import sqlite3
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from vanna.chromadb import ChromaDB_VectorStore
from vanna.ollama import Ollama
from vanna.openai import OpenAI_Chat


class OpenaiChromaVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

    def get_sql_prompt_(
        self,
        initial_prompt: str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        gen_sql: bool,
        **kwargs,
    ):
        """
        Example:
        ```python
        vn.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?", "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        if not initial_prompt:
            initial_prompt = (
                f"You are a {self.dialect} expert. "
                + "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "
            )

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        if gen_sql:
            initial_prompt += (
                "===Response Guidelines \n"
                "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
                "2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
                "3. If the provided context is insufficient, please explain why it can't be generated. \n"
                "4. Please use the most relevant table(s). \n"
                "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
                "6. Column names are traditional chinese (zh-TW), neither simplified chinese nor english is allowed. \n"
            )
        else:
            initial_prompt += (
                "===Response Guidelines \n"
                "1. generate result based on the information related. \n"
                "2. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
                "3. Response only in traditional chinese (zh-TW). \n"
            )
        if "mpos" in question.lower():
            "4. Add '好的，已幫您紀錄完成' after the answer if and only if the question mentioned 'mPOS'."

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log

    def generate_sql_safe(
        self, question: str, allow_llm_to_see_data=False, **kwargs
    ) -> str:
        """
        Example:
        ```python
        vn.generate_sql("What are the top 10 customers by sales?")
        ```

        Uses the LLM to generate a SQL query that answers a question. It runs the following methods:

        - [`get_similar_question_sql`][vanna.base.base.VannaBase.get_similar_question_sql]

        - [`get_related_ddl`][vanna.base.base.VannaBase.get_related_ddl]

        - [`get_related_documentation`][vanna.base.base.VannaBase.get_related_documentation]

        - [`get_sql_prompt`][vanna.base.base.VannaBase.get_sql_prompt]

        - [`submit_prompt`][vanna.base.base.VannaBase.submit_prompt]


        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt_(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            gen_sql=True,
            **kwargs,
        )

        self.log(title="SQL Prompt", message=prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        if "intermediate_sql" in llm_response:
            if not allow_llm_to_see_data:
                return "The LLM is not allowed to see the data in your database. Your question requires database introspection to generate the necessary SQL. Please set allow_llm_to_see_data=True to enable this."

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    prompt = self.get_sql_prompt_(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list
                        + [
                            f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n"
                            + df.to_markdown()
                        ],
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message=prompt)
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    return f"Error running intermediate SQL: {e}"

        try:
            ext_llm_response = self.extract_sql(llm_response)
            sql_result = self.run_sql(ext_llm_response)

            print(sql_result)

            inst_result = self.submit_prompt(
                [
                    self.system_message(
                        # "You are a consultant in a insurance company. Given question and prev_result, you add brief instruction before or after the prev_result to help the user understand better."
                        # "===Response Guidelines \n"
                        # "1. The prev_result must be shown and do not revise the content. \n"
                        # "2. If the question contains more than one question and the prev_result is not null, then assume the previous question is positive and answer. \n"
                        # "3. Give advise to the person in the user's question, not the user itself. \n"
                        # "4. Response only in traditional chinese (zh-TW)"
                        # "5. Do not repeat the question when you response."
                        "You are a consultant in a insurance company. Given question and prev_result, you add brief instruction before or after the prev_result to help the user understand better."
                        "NEVER REVISE THE prev_result. The information is a must to shown. "
                        "Response only in traditional chinese (zh-TW). "
                        # "Do not repeat the question when you response. "
                        # "Give advise to the person in the user's question, not the user itself. "
                    )
                ]
                + [
                    self.user_message(
                        f"question: {question}, prev_result: {sql_result.to_markdown()}"
                    )
                ]
            )

            return inst_result

        except:
            initial_prompt = (
                f"You are a insurance expert. "
                + "Please help to generate response to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "
            )
            prompt_wo_sql = self.get_sql_prompt_(
                initial_prompt=initial_prompt,
                question=question,
                question_sql_list=[],
                ddl_list=ddl_list,
                doc_list=doc_list,
                gen_sql=False,
                **kwargs,
            )

            self.log(
                title="SQL execution failed, will use valinna rag",
                message=prompt_wo_sql,
            )
            llm_response = self.submit_prompt(prompt_wo_sql, **kwargs)
            self.log(title="LLM Response", message=llm_response)

            return llm_response

    def genrate_rag(self, question: str, **kwargs) -> str:
        initial_prompt = (
            f"You are a insurance expert. "
            + "Please help to generate response to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "
        )
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt_(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=[],
            ddl_list=[],
            doc_list=doc_list,
            gen_sql=False,
            **kwargs,
        )
        print(f"{prompt=}")
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        return llm_response


class OllamaChromaVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

    def get_sql_prompt_(
        self,
        initial_prompt: str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        gen_sql: bool,
        **kwargs,
    ):
        """
        Example:
        ```python
        vn.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?", "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        if not initial_prompt:
            initial_prompt = (
                f"You are a {self.dialect} expert. "
                + "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "
            )

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        if gen_sql:
            initial_prompt += (
                "===Response Guidelines \n"
                "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
                "2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
                "3. If the provided context is insufficient, please explain why it can't be generated. \n"
                "4. Please use the most relevant table(s). \n"
                "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
                "6. Column names are traditional chinese (zh-TW), neither simplified chinese nor english is allowed. \n"
            )
        else:
            initial_prompt += (
                "===Response Guidelines \n"
                "1. generate result based on the information related. \n"
                "2. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
                "3. Response only in traditional chinese (zh-TW). \n"
            )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log

    def generate_sql_safe(
        self, question: str, allow_llm_to_see_data=False, **kwargs
    ) -> str:
        """
        Example:
        ```python
        vn.generate_sql("What are the top 10 customers by sales?")
        ```

        Uses the LLM to generate a SQL query that answers a question. It runs the following methods:

        - [`get_similar_question_sql`][vanna.base.base.VannaBase.get_similar_question_sql]

        - [`get_related_ddl`][vanna.base.base.VannaBase.get_related_ddl]

        - [`get_related_documentation`][vanna.base.base.VannaBase.get_related_documentation]

        - [`get_sql_prompt`][vanna.base.base.VannaBase.get_sql_prompt]

        - [`submit_prompt`][vanna.base.base.VannaBase.submit_prompt]


        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt_(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            gen_sql=True,
            **kwargs,
        )
        self.log(title="SQL Prompt", message=prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        if "intermediate_sql" in llm_response:
            if not allow_llm_to_see_data:
                return "The LLM is not allowed to see the data in your database. Your question requires database introspection to generate the necessary SQL. Please set allow_llm_to_see_data=True to enable this."

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    prompt = self.get_sql_prompt_(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list
                        + [
                            f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n"
                            + df.to_markdown()
                        ],
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message=prompt)
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    return f"Error running intermediate SQL: {e}"

        try:
            ext_llm_response = self.extract_sql(llm_response)
            sql_result = self.run_sql(ext_llm_response)

            print(sql_result)

            inst_result = self.submit_prompt(
                [
                    self.system_message(
                        # "You are a consultant in a insurance company. Given question and prev_result, you add brief instruction before or after the prev_result to help the user understand better."
                        # "===Response Guidelines \n"
                        # "1. The prev_result must be shown and do not revise the content. \n"
                        # "2. If the question contains more than one question and the prev_result is not null, then assume the previous question is positive and answer. \n"
                        # "3. Give advise to the person in the user's question, not the user itself. \n"
                        # "4. Response only in traditional chinese (zh-TW)"
                        # "5. Do not repeat the question when you response."
                        "You are a consultant in a insurance company. Given question and prev_result, you add brief instruction before or after the prev_result to help the user understand better."
                        "NEVER REVISE THE prev_result. The information is a must to shown. "
                        "Response only in traditional chinese (zh-TW). "
                        # "Do not repeat the question when you response. "
                        # "Give advise to the person in the user's question, not the user itself. "
                    )
                ]
                + [
                    self.user_message(
                        f"question: {question}, prev_result: {sql_result.to_markdown()}"
                    )
                ]
            )

            return inst_result

        except:
            initial_prompt = (
                f"You are a insurance expert. "
                + "Please help to generate response to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "
            )
            prompt_wo_sql = self.get_sql_prompt_(
                initial_prompt=initial_prompt,
                question=question,
                question_sql_list=[],
                ddl_list=ddl_list,
                doc_list=doc_list,
                gen_sql=False,
                **kwargs,
            )

            self.log(
                title="SQL execution failed, will use valinna rag",
                message=prompt_wo_sql,
            )
            llm_response = self.submit_prompt(prompt_wo_sql, **kwargs)
            self.log(title="LLM Response", message=llm_response)

            return llm_response

    def genrate_rag(self, question: str, **kwargs) -> str:
        initial_prompt = (
            f"You are a insurance expert. "
            + "Please help to generate response to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "
        )
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt_(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=[],
            ddl_list=[],
            doc_list=doc_list,
            gen_sql=False,
            **kwargs,
        )
        print(f"{prompt=}")
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        return llm_response


class VannaSQLLLM(LLM):
    llm: OllamaChromaVanna | OpenaiChromaVanna

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.llm.generate_sql_safe(prompt)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for char in self.llm.generate_sql_safe(prompt):
            # for char in self.llm.run_sql(self.llm.generate_sql(prompt)).to_markdown():
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomSQLChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "customSQL"


class VannaRAGLLM(LLM):
    llm: OllamaChromaVanna | OpenaiChromaVanna

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.llm.genrate_rag(prompt)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for char in self.llm.genrate_rag(prompt):
            # for char in self.llm.run_sql(self.llm.generate_sql(prompt)).to_markdown():
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomRAGChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "customRAG"
