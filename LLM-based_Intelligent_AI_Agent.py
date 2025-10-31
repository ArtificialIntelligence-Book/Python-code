"""
LLM-Based Agent Framework in Python

This example illustrates the key concepts of an intelligent AI agent architecture:
- Processing user requests (natural language understanding).
- Decision-making (selecting appropriate tool/actions).
- Executing tasks via multiple tools (tool orchestration).
- Adaptive behavior by managing context and state.

Key Components:
1. LLMInterface: Simulates interaction with a Large Language Model (LLM) for understanding and generation.
2. ToolBase: Abstract base class for tools the agent can use.
3. Specific tools: Example tools like Calculator, WebSearch.
4. Agent: Core agent coordinating understanding, planning, and tool execution.
5. Demonstration of agent processing user requests, deciding which tool(s) to use, and returning results.

Note:
- The LLM here is simulated; replace with actual API calls (e.g., OpenAI GPT).
- Tool implementations are simplified for demonstration.

"""

import random
import re

# === 1. LLM Interface ===
class LLMInterface:
    """
    Simulates an LLM that processes text and produces responses.
    In practice, this would be replaced by API calls to models like GPT-4.
    """

    def __init__(self):
        pass

    def understand_intent(self, user_input):
        """
        Simulate intent recognition from user input.
        Returns a dict with 'intent' and 'entities'.
        """
        # Very simple keyword-based intent recognition for demo
        user_input_lower = user_input.lower()
        if any(word in user_input_lower for word in ['calculate', 'compute', 'what is', 'evaluate']):
            return {'intent': 'calculate', 'entities': {}}
        elif any(word in user_input_lower for word in ['search', 'find', 'look up']):
            return {'intent': 'web_search', 'entities': {}}
        else:
            return {'intent': 'chat', 'entities': {}}

    def generate_response(self, prompt):
        """
        Simulate generating an LLM response.
        """
        canned_responses = [
            "I'm here to help! What would you like to do next?",
            "Could you please clarify your request?",
            "Processing your request now.",
            "Here's what I found for you."
        ]
        return random.choice(canned_responses)

# === 2. Tool Base Class ===
class ToolBase:
    """
    Abstract base class for tools.
    """

    def name(self):
        raise NotImplementedError

    def execute(self, query):
        """
        Execute the tool's function with the given query.
        Returns a string result.
        """
        raise NotImplementedError

# === 3. Specific Tools ===

class CalculatorTool(ToolBase):
    def name(self):
        return "Calculator"

    def execute(self, query):
        """
        Evaluate simple arithmetic expressions safely.
        """
        try:
            # Extract expression after keywords
            expr = query.lower()
            expr = re.sub(r'[^0-9\.\+\-\*\/\(\) ]', '', expr)
            if not expr.strip():
                return "No valid expression found to calculate."

            # Evaluate expression securely
            result = eval(expr, {"__builtins__": {}})
            return f"The result is: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"

class WebSearchTool(ToolBase):
    def name(self):
        return "WebSearch"

    def execute(self, query):
        """
        Simulate a web search returning top results.
        """
        # In real system, integrate with search API like Google Custom Search or Bing
        return f"Simulated search results for '{query}':\n1. Result A\n2. Result B\n3. Result C"

# === 4. Agent Core ===
class LLMBasedAgent:
    def __init__(self):
        self.llm = LLMInterface()
        self.tools = {
            'calculate': CalculatorTool(),
            'web_search': WebSearchTool(),
        }
        self.context = []  # History of interactions for adaptive behavior

    def process_request(self, user_input):
        """
        Main method to process user requests.
        Steps:
        - Understand intent via LLM.
        - Decide which tool(s) to use.
        - Execute tool(s).
        - Generate response combining tool outputs.
        """

        # Step 1: Understand intent
        intent_data = self.llm.understand_intent(user_input)
        intent = intent_data.get('intent')
        print(f"[DEBUG] Recognized intent: {intent}")

        # Step 2: Decision making and tool selection
        if intent in self.tools:
            tool = self.tools[intent]
            tool_result = tool.execute(user_input)
            response = f"Using {tool.name()}:\n{tool_result}"
        else:
            # Default chat behavior
            response = self.llm.generate_response(user_input)

        # Step 3: Update context
        self.context.append({'user': user_input, 'agent': response})

        return response

    def show_context(self):
        """
        Display conversation history.
        """
        print("=== Conversation History ===")
        for i, turn in enumerate(self.context):
            print(f"User: {turn['user']}")
            print(f"Agent: {turn['agent']}\n")

# === 5. Demo Usage ===
if __name__ == "__main__":
    agent = LLMBasedAgent()

    print("Welcome to the LLM-Based Agent Demo!")
    print("Type 'exit' to quit or 'history' to see conversation history.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'history':
            agent.show_context()
            continue

        response = agent.process_request(user_input)
        print(f"Agent: {response}\n")