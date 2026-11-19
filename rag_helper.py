
INSTRUCTIONS = """
You are a question-answering assistant for the LLM Zoomcamp course.

Your task is to answer questions using ONLY the information
provided in the context.

Rules:
1. Treat the context as the only source of truth.
2. Do not use your pretrained knowledge or external knowledge.
3. Do not make assumptions or invent information.
4. If the context does not contain enough information to answer
   the question, respond exactly with:
   I don't know
5. Do not answer questions unrelated to the course.
6. If the answer is available in the context, provide a clear
   and concise answer based on that information.
"""
USER_PROMP_TEMPLATE = """
Question :
{question}

Context:
{context}
""".strip()

class RAGBase:

    def __init__(
            self,
            index,
            llm_client,
            instructions = INSTRUCTIONS,
            course = 'llm-zoomcamp',
            prompt_template = USER_PROMP_TEMPLATE,
            model = 'gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template  = prompt_template
        self.model = model




    def search(self, query, num_results = 5):
        boost_dict = {'question': 3.0, 'section': 0.5}
        filter_dict = {'course': self.course}
        return self.index.search(
            query, 
            boost_dict=boost_dict,
            filter_dict = filter_dict,
            num_results = num_results
        )



    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['section'])
            lines.append('Q: ' + doc['question'])
            lines.append('A: ' + doc['answer'])
            lines.append('')
        return '\n'.join(lines).strip()


    def build_prompt(self, query, search_results):
        context= self.build_context(search_results)
        prompt = self.prompt_template.format(question=query, context=context)
        return prompt.strip()

    def llm(self, prompt):
        input_messages = [
        {'role': 'system', 'content': self.instructions},
        {'role': 'user', 'content': prompt}]
        response = self.llm_client.responses.create(
            model = self.model,
            input = input_messages
        )
        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt=self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer