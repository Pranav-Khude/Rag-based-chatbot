import os
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphCypherQAChain


NEO4J_URI="neo4j+s://f7cd7ce0.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "orE_h31a3__FAaegaNjUikeYudl5yeABO8u31ZOH-CU"
GROQ_API_KEY = "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"

# Initialize Neo4j graph
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

# Grok API key (replace with your actual key)
groq_api_key = GROQ_API_KEY

# Initialize Grok LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Sample medical knowledge
SAMPLE_MEDICAL_KNOWLEDGE = [
    "Hypertension is a condition characterized by high blood pressure. It is commonly treated with medications such as ACE Inhibitors and Diuretics. Procedures like blood pressure monitoring are used to manage it. Risk factors include obesity, smoking, and stress. It is often related to Heart Disease and Chronic Kidney Disease.",
    "Heart Disease affects the heart and blood vessels. It is treated with medications like Statins and Beta Blockers. Procedures such as coronary artery bypass and angioplasty are common interventions. Risk factors include high cholesterol, Hypertension, and smoking. It can lead to complications like heart failure.",
    "Diabetes is a metabolic disorder involving high blood sugar levels. It is managed with medications like Insulin and Metformin. Glucose monitoring is a key procedure for control. Risk factors include obesity, family history, and sedentary lifestyle. It is a comorbidity with Heart Disease.",
    "Chronic Kidney Disease involves the gradual loss of kidney function. It is treated with medications like Diuretics and managed through procedures like dialysis. Risk factors include Hypertension and Diabetes. It can result in fluid retention and requires careful monitoring."
]

# Convert text to Documents
documents = [Document(page_content="\n".join(SAMPLE_MEDICAL_KNOWLEDGE))]

# Initialize LLMGraphTransformer with Grok
llm_transformer = LLMGraphTransformer(llm=llm)

# Convert documents to graph documents
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Add graph documents to Neo4j
graph.add_graph_documents(graph_documents)

# Refresh schema and debug
graph.refresh_schema()
print("Graph Schema:")
print(graph.schema)

# Debug: Check node labels and sample nodes
labels_query = "CALL db.labels() YIELD label RETURN label"
nodes_query = "MATCH (n) RETURN labels(n) AS labels, n LIMIT 5"
labels = graph.query(labels_query)
nodes = graph.query(nodes_query)
print("Node Labels:", labels)
print("Sample Nodes:", nodes)

CYPHER_PROMPT = """
Given the following question: {question}
Generate a Cypher query to retrieve the answer from a Neo4j graph with nodes labeled as:
- 'Condition' for diseases (property: name)
- 'Medication' for treatments (property: name)
- 'Procedure' for interventions (property: name)
- 'RiskFactor' for risk factors (property: name)
Relationships include:
- :TREATED_WITH (Condition to Medication)
- :MANAGED_BY (Condition to Procedure)
- :RISK_FACTOR_FOR (RiskFactor to Condition)
- :RELATED_TO (Condition to Condition)
Return the query as a string.
"""

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=CYPHER_PROMPT
)

# Test queries
queries = [
    "What medications are used for Hypertension?",
    "What are the risk factors for Diabetes?",
    "How is Heart Disease related to Hypertension?",
    "What procedures are associated with Chronic Kidney Disease?"
]

for query in queries:
    print(f"\nQuery: {query}")
    response = chain.invoke({"query": query})
    print(f"Response: {response['result']}")

# Optional: Clear the graph if you want to rerun
graph.query("MATCH (n) DETACH DELETE n")