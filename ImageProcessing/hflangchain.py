from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

eu_llm = HuggingFaceHub(
    repo_id="gpt2",
    model_kwargs = {'temperature' : 0.1, 'max_length': 256}
    )

prompt = PromptTemplate(
    input_variables=["text"],
    template="extract key value pairs from text given text here, for example : model 67ght, size large etc., {text} "
)

chain = LLMChain(prompt=prompt, llm = eu_llm, verbose=True)
print(chain.run("model 7hgy5 age 18 size large"))


