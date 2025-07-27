import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from flask_cors import CORS

# Assume the AI helper functions (ai_helper, math_helper, social_helper,
# generate_math_structured_response, generate_social_structured_response,
# classify_question, enhanced_ai_helper) are in the same file or imported.
# For this example, we will include them.

# --- Start of Helper Functions (Copy/Paste from your provided code) ---

def ai_helper(question: str) -> dict:
    """Retrieves the data using gemini AI."""
    try:
        client = genai.Client(vertexai=True, project='nestbees', location='global')
        print(f"question: {question}")
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents = question
        )
        return {
            "status": "success",
            "report": response.text
        }
    except Exception as e:
        return {
            "status": "error",
            "report": f"Error in general AI helper: {str(e)}"
        }


def math_helper(question: str) -> dict:
    """Retrieves the data from RAG Model regarding math and generates structured response."""
    try:
        client = genai.Client(
            vertexai=True, project='nestbees', location='global'
        )

        model = "gemini-2.5-flash-lite"
        contents = [
            types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=question)
            ]
            )
        ]
        tools = [
            types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                rag_resources=[
                    types.VertexRagStoreRagResource(
                    rag_corpus="projects/nestbees/locations/us-central1/ragCorpora/4611686018427387904"
                    )
                ],
                )
            )
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature = 1,
            top_p = 0.95,
            max_output_tokens = 5000,
            safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
            )],
            tools = tools,
            system_instruction=[types.Part.from_text(text="""You are a math teacher's assistant. Explain every mathematical topic in detail with step-by-step solutions. Use simple, layman language that students can easily understand. Include real-world examples and applications. Break down complex concepts into digestible parts.""")],
            thinking_config=types.ThinkingConfig(
                # include_thoughts=True,
                thinking_budget=-1,
            ),
        )

        response_text = ""

        print(f"math question: {question}")

        # Get RAG response
        for chunk in client.models.generate_content_stream(
                model=model,
                contents= contents,
                config=generate_content_config,
                ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            if chunk.text is not None:
                response_text += chunk.text

        # Generate enhanced structured response
        try:
            enhanced_response = generate_math_structured_response(question, response_text)
            # enhanced_response = response_text
            return {
                "status": "success",
                "report": enhanced_response
            }
        except Exception as e:

            return {
                "status": "error",
                "report": f"Error generating structured response: {str(e)}"
            }
    except Exception as e:
        return {
            "status": "error",
            "report": f"Error in math helper: {str(e)}"
        }

def social_helper(question: str) -> dict:
    """Retrieves the data from RAG Model regarding social studies and generates structured response."""
    try:
        client = genai.Client(
            vertexai=True,
            project="nestbees",
            location="global",
        )

        model = "gemini-2.5-flash-lite"
        contents = [
            types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=question)
            ]
            )
        ]

        # Replace with your social studies RAG corpus ID
        tools = [
            types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                rag_resources=[
                    types.VertexRagStoreRagResource(
                    rag_corpus="projects/nestbees/locations/us-central1/ragCorpora/4611686018427387904"  # Replace with your social studies corpus ID
                    )
                ],
                )
            )
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature = 1,
            top_p = 0.95,
            max_output_tokens = 5000,
            safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
            )],
            tools = tools,
            system_instruction=[types.Part.from_text
(text="""You are a social studies teacher's assistant. Explain every historical, geographical, political, and cultural topic in detail using storytelling and engaging narratives. Use simple, layman language that students can easily understand. Connect historical events to modern-day relevance. Make complex social concepts relatable through real-world examples and analogies.""")],
            thinking_config=types.ThinkingConfig(
#                 include_thoughts=True,
                thinking_budget=-1,
            ),
        )

        response_text = ""

        print(f"social studies question: {question}")

        # Get RAG response
        for chunk in client.models.generate_content_stream(
                model=model,
                contents= contents,
                config=generate_content_config,
                ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            if chunk.text is not None:
                response_text += chunk.text

        # Generate enhanced structured response
        try:
            enhanced_response = generate_social_structured_response(question, response_text)
            return {
                "status": "success",
                "report": enhanced_response
            }
        except Exception as e:
            return {
                "status": "error",
                "report": f"Error generating structured response: {str(e)}"
            }
    except Exception as e:
        return {
            "status": "error",
            "report": f"Error in social studies helper: {str(e)}"
        }


def generate_math_structured_response(question: str, rag_response: str) -> dict:
    """Generate structured response for math questions with script, prompt, and video keys."""
    try:
        client = genai.Client(vertexai=True, project='nestbees', location='global')

        # Generate detailed script explanation
        script_prompt = f"""
        Based on this math question: "{question}"
        And this detailed mathematical explanation: "{rag_response}"

        Create a comprehensive teaching script that:
        1. Explains the mathematical concept in simple, everyday language
        2. Provides step-by-step solution with clear reasoning
        3. Includes real-world examples and applications
        4. Uses analogies to make complex concepts understandable
        5. Encourages students and builds confidence

        Write this as if you're speaking directly to a student, using "you" and conversational tone.
        Make it engaging and easy to follow.
        """

        # script_response = client.models.generate_content(
        #     model="gemini-2.5-flash",
        #     contents=script_prompt
        # )

        # Create the structured response
        structured_response = {
            # "script": script_response.text,
            "script": rag_response,
            "prompt": f"{question}\n\nSolve this step-by-step, showing all work and explaining each mathematical concept used. Include formulas, calculations, and verify your answer.",
            "video": False  # Math questions don't need video as per requirement
        }

        return structured_response
    except Exception as e:
        print(f"Error generating math structured response: {e}")
        # Return a basic structure on error
        return {
            "script": "Could not generate a detailed script.",
            "prompt": f"{question}\n\nCould not generate a detailed prompt.",
            "video": False
        }


def generate_social_structured_response(question: str, rag_response: str) -> dict:
    """Generate structured response for social studies questions with script, prompt, and video keys."""
    try:
        client = genai.Client(vertexai=True, project='nestbees', location='global')

        # Generate detailed script explanation
        script_prompt = f"""
        Based on this social studies question: "{question}"
        And this detailed historical/social explanation: "{rag_response}"

        Create a comprehensive teaching script that:
        1. Tells the story in an engaging, narrative style
        2. Explains historical context and significance in simple terms
        3. Connects past events to modern-day relevance
        4. Uses storytelling techniques to make it memorable
        5. Explains complex social/political concepts using everyday analogies
        6. Encourages curiosity about history and society

        Write this as if you're telling an interesting story to a student, using vivid descriptions and conversational tone.
        Make it feel like an engaging documentary narrative.
        """

        # script_response = client.models.generate_content(
        #     model="gemini-2.5-flash",
        #     contents=script_prompt
        # )

        # Create the structured response
        structured_response = {
            "script": rag_response,
            "prompt": f"{question}\n\nProvide a comprehensive answer covering historical context, cultural significance, key figures involved, causes and effects, and modern relevance. Use storytelling to make it engaging.",
            "video": True  # Social studies questions need video as per requirement
        }

        return structured_response
    except Exception as e:
        print(f"Error generating social structured response: {e}")
        # Return a basic structure on error
        return {
            "script": "Could not generate a detailed script.",
            "prompt": f"{question}\n\nCould not generate a detailed prompt.",
            "video": False
        }

def classify_question(question: str) -> str:
    """Classify if a question is about math or social studies using AI."""

    client = genai.Client(vertexai=True, project='nestbees', location='global')

    classification_prompt = f"""
    Classify this educational question into one of these categories: "math", "social_studies", or "general"

    Question: "{question}"

    Guidelines:
    - "math" for: arithmetic, algebra, geometry, calculus, statistics, equations, formulas, calculations, mathematical concepts
    - "social_studies" for: history, government, politics, geography, culture, civics, economics, civilizations, historical events, countries, leaders
    - "general" for: everything else that doesn't clearly fit math or social studies

    Respond with only one word: "math", "social_studies", or "general"
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=classification_prompt
        )

        classification = response.text.strip().lower()

        if "math" in classification:
            return "math"
        elif "social" in classification:
            return "social_studies"
        else:
            return "general"

    except Exception as e:
        print(f"Classification error: {e}")
        # Fallback to keyword-based classification
        question_lower = question.lower()

        math_keywords = ['calculate', 'solve', 'equation', 'formula', 'algebra', 'geometry', 'math', 'arithmetic']
        social_keywords = ['history', 'government', 'politics', 'democracy', 'president', 'war', 'culture', 'geography']

        math_score = sum(1 for keyword in math_keywords if keyword in question_lower)
        social_score = sum(1 for keyword in social_keywords if keyword in question_lower)

        if math_score > social_score:
            return "math"
        elif social_score > 0:
            return "social_studies"
        else:
            return "general"

def enhanced_ai_helper(question: str) -> dict:
    """Enhanced AI helper that routes questions appropriately and returns structured responses."""

    # Classify the question
    question_type = classify_question(question)

    print(f"Question classified as: {question_type}")

    if question_type == "math":
        answer = math_helper(question)
        print(f"answer : {answer}")
        return answer
    elif question_type == "social_studies":
        return social_helper(question)
    else:
        # For general questions, use the original ai_helper
        return ai_helper(question)

# --- End of Helper Functions ---


app = Flask(__name__)
CORS(app, origins=["*"])

@app.route("/ask_ai", methods=["POST"])
def ask_ai_endpoint():
    """API endpoint to receive a question and return a structured AI response."""
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"status": "error", "message": "Invalid request. 'question' field is required."}), 400

    question = data["question"]

    # Use the enhanced_ai_helper to get the response
    response = enhanced_ai_helper(question)

    # Return the structured response as JSON
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))