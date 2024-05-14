from flask import Flask, request, jsonify
import os
import re
from openai import OpenAI
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/', methods=['GET'])
def index():
    return jsonify(message='Welcome to Interview AI!')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.get_json()
        sent_question_ids = data.get('sent_question_ids', [])
        candidate_answers = data.get('candidate_answers', {})
        questions = data.get('questions', {})
        print(sent_question_ids, candidate_answers, questions)

        responses = {}
        for question_id in sent_question_ids:
            question = questions.get(str(question_id), '')
            candidate_answer = candidate_answers.get(str(question_id), '')
            if question and candidate_answer:  # Ensure both question and answer are present
                result = evaluate_answer(question, candidate_answer)
                responses[question_id] = result
            else:
                responses[question_id] = {"feedback": "Question or answer missing.", "score": None}

        return jsonify(responses)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def evaluate_answer(question, candidate_answer):
    prompt = f"""Please evaluate the following answer to the question '{question}'. Analyze the depth of understanding and accuracy demonstrated by the candidate's response. Provide a detailed summary that includes key insights, any discrepancies from expected knowledge. At the end of your evaluation, explicitly state the confidence score in the format 'Score: X.X', where X.X is a decimal number between 0 and 1 representing how well the candidate answered the question.
Answer: {candidate_answer}"""
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=300,
        temperature=0.5,
    )

    completion_text = response.choices[0].text
    print(completion_text)

    # Extract score using a specific regex pattern
    score_match = re.search(r"Score:\s*(\d\.\d+)", completion_text, re.IGNORECASE)
    score = float(score_match.group(1)) if score_match else "Score not found"

    return {
        "feedback": completion_text.strip(),
        "score": score
    }

if __name__ == '__main__':
    app.run(debug=True)
