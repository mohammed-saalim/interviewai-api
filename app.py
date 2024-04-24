#version1.1


from flask import Flask, request, jsonify
import re
from openai import OpenAI
from flask_cors import CORS
import pdb
import os


app = Flask(__name__)
CORS(app)

# Set up OpenAI API key
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

# Dictionary of answers with question IDs as keys
answers = {
    1: "OOP stands for Object-Oriented Programming. It's a programming paradigm that revolves around the concept of objects which can contain data in the form of fields (attributes or properties), and code in the form of procedures (methods or functions).",
    2: "Polymorphism is one of the fundamental concepts in object-oriented programming (OOP). It refers to the ability of different objects to respond in different ways to the same message or method invocation. In other words, polymorphism allows objects of different classes to be treated as objects of a common superclass, enabling a single interface to represent multiple underlying forms.",
    3: "Inheritance is a fundamental concept in object-oriented programming (OOP) that allows a new class (subclass or derived class) to inherit properties and behaviors (methods and fields) from an existing class (superclass or base class). This means that the subclass can reuse code from the superclass and extend it to add new functionality or override existing behavior.",
    4: "Data abstraction is a fundamental principle in software engineering and computer science, particularly in the context of object-oriented programming (OOP). It refers to the concept of representing essential features without including the background details or explanations.",
    5: "Data encapsulation, also known simply as encapsulation, is a fundamental concept in object-oriented programming (OOP) that involves bundling data (attributes or properties) and methods (functions or procedures) that operate on that data within a single unit, typically a class. Encapsulation hides the internal state of an object from the outside world and only exposes a public interface through which external code can interact with the object.",
    # Add more answers as needed
}

def evaluate_similarity(reference_answer, candidate_answer):
    prompt = f"We are assessing candidates' understanding of technical concepts based on their responses in an interview. Below is the original answer we consider as a standard for understanding a specific technical question. Following that, you'll find a candidate's response to the same question. Your task is to analyze the candidate's response in comparison to the original answer and evaluate the similarity in context and understanding. Provide a summary that includes whether the candidate has grasped the key concepts, the depth of their understanding, and any significant discrepancies or additional insights they've offered compared to the original answer. Also, Give a confidence/similarity numerical score comparing the reference answer with Candidate's answer ranging from approximately 0 to 1. Store the decimal score in variable 'Score'. Reference answer: {reference_answer}\nCandidate answer: {candidate_answer}\n"
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=3000,
        temperature=0.5,
    )

    completion_text = response.choices[0].text
    print(completion_text)
    
    return completion_text

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        count=0
        data = request.get_json()
        print(data)
        sent_question_ids = data.get('sent_question_ids', [])
        candidate_answers = data.get('candidate_answers', {})

        responses = {}

        for question_id in sent_question_ids:
            count=count+1
            reference_answer = answers.get(question_id, "")  # Get the reference answer for the question ID
            candidate_answer = candidate_answers.get(str(question_id), '')  # Get the candidate answer for the question ID
            if count == 4:
                break
            # pdb.set_trace()
            completion_text = evaluate_similarity(reference_answer, candidate_answer)
            responses[question_id] = completion_text


        return jsonify(responses)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500  

    #     full_answer = completion_text

    #     match = re.search(r'Score: (\d+\.\d+)', completion_text)
    #     if match:
    #         # confidence_score = float(match.group(1))
    #         # return jsonify({'confidence_score': confidence_score, 'feedback': full_answer})
    #         return jsonify({'feedback': full_answer})
    #     else:
    #         return jsonify({'error': 'Confidence score not found'}), 500
        
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500    

if __name__ == '__main__':
    app.run(debug=True)
   