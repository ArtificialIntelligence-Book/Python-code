"""
Simple Question Answering System Using a Knowledge Graph and a Large Language Model (LLM)

This example demonstrates a hybrid QA system combining:
- A small Knowledge Graph (KG) represented as a dictionary of triples.
- A Large Language Model (LLM) simulated via OpenAI GPT API or a placeholder function.
- KG is used for structured factual queries.
- LLM is used to interpret questions and generate answers by combining KG info and language understanding.

Key components:
1. KnowledgeGraph class: stores triples and performs simple query matching.
2. SimpleLLM class: simulates an LLM that can generate answers given context.
3. QA_System class: integrates KG querying and LLM answering.
4. Demo usage with example questions.

Note:
- For demonstration, the LLM part here is a placeholder function returning canned responses.
- Replace `SimpleLLM.generate_answer` with real LLM API calls (e.g., OpenAI GPT).
- KG querying is basic; real systems use graph databases and SPARQL or embedding-based retrieval.

Dependencies:
- None for the demo code below.
- For real LLM: `openai` or other SDKs.

"""

# === 1. Knowledge Graph ===
class KnowledgeGraph:
    def __init__(self):
        # Store triples as list of (subject, predicate, object)
        self.triples = []

    def add_triple(self, subject, predicate, object_):
        self.triples.append((subject.lower(), predicate.lower(), object_.lower()))

    def query(self, subject=None, predicate=None, object_=None):
        """
        Simple pattern matching query on triples.
        None means wildcard.
        Returns list of matching triples.
        """
        results = []
        for (s, p, o) in self.triples:
            if (subject is None or s == subject.lower()) and \
               (predicate is None or p == predicate.lower()) and \
               (object_ is None or o == object_.lower()):
                results.append((s, p, o))
        return results

    def facts_about(self, entity):
        """
        Return all triples where the entity is subject or object.
        """
        entity = entity.lower()
        facts = [t for t in self.triples if t[0] == entity or t[2] == entity]
        return facts

# === 2. Simple LLM Simulator ===
class SimpleLLM:
    def __init__(self):
        pass

    def generate_answer(self, question, knowledge_context):
        """
        Generate an answer given a question and knowledge graph context.
        Here, we simulate by returning a combined string.
        Replace with real LLM API calls in practice.
        """
        if not knowledge_context:
            return "I'm sorry, I don't have information about that."

        # Format knowledge context nicely
        facts = "\n".join([f"{s.title()} {p} {o.title()}" for s, p, o in knowledge_context])

        answer = (
            f"Based on the knowledge graph, here are some relevant facts:\n{facts}\n\n"
            f"Answer to your question: {question}"
        )
        return answer

# === 3. Question Answering System ===
class QA_System:
    def __init__(self, kg, llm):
        self.kg = kg
        self.llm = llm

    def answer_question(self, question):
        """
        Process the question: extract entities, query KG, and generate answer.
        Here, entity extraction is simulated by keyword matching.
        """

        # VERY simple entity extraction: check for known entities in KG triples
        question_lower = question.lower()
        known_entities = set()
        for s, p, o in self.kg.triples:
            known_entities.add(s)
            known_entities.add(o)

        # Find entities mentioned in question
        entities_in_question = [e for e in known_entities if e in question_lower]

        # If no entity found, no KG context
        if not entities_in_question:
            knowledge_context = []
        else:
            # Aggregate facts about found entities
            knowledge_context = []
            for ent in entities_in_question:
                knowledge_context.extend(self.kg.facts_about(ent))

            # Deduplicate facts
            knowledge_context = list(set(knowledge_context))

        # Use LLM to generate answer using KG context
        answer = self.llm.generate_answer(question, knowledge_context)
        return answer

# === 4. Demo Usage ===
if __name__ == "__main__":
    # Create knowledge graph and add triples
    kg = KnowledgeGraph()
    kg.add_triple("Albert Einstein", "born in", "Ulm")
    kg.add_triple("Albert Einstein", "profession", "physicist")
    kg.add_triple("Albert Einstein", "won", "Nobel Prize in Physics")
    kg.add_triple("Nobel Prize in Physics", "awarded for", "contributions to theoretical physics")
    kg.add_triple("Isaac Newton", "born in", "Woolsthorpe")
    kg.add_triple("Isaac Newton", "profession", "physicist")
    kg.add_triple("Isaac Newton", "discovered", "gravity")

    # Initialize simple LLM simulator
    llm = SimpleLLM()

    # Initialize QA system
    qa_system = QA_System(kg, llm)

    # Example questions
    questions = [
        "Where was Albert Einstein born?",
        "What did Isaac Newton discover?",
        "Who won the Nobel Prize in Physics?",
        "Tell me about Marie Curie."
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = qa_system.answer_question(q)
        print(f"Answer:\n{answer}")