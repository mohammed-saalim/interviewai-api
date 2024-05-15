# AI Interview API

## Introduction
This repository hosts the Flask API for the AI Interview Platform, a powerful tool designed to automate the interview process using AI. It evaluates interview responses by leveraging OpenAI's GPT models to provide scores and detailed feedback. This project was demonstrated at the OraHacks hackathon.

## API Deployment
The API is currently deployed on Heroku and can be accessed here: [AI Interview API](https://interviewaiapi-2194adb0afda.herokuapp.com/). This deployment serves as the backend for the interview platform, handling requests and processing data.

## Key Endpoint
- **/evaluate**: A POST endpoint that accepts interview answers, communicates with OpenAI for processing, and returns detailed feedback and scores to the front-end application.

## Setup and Local Development
To set up and run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohammed-saalim/orahacks-api.git
   cd orahacks-api
2. **Install dependencies**:
   pip install -r requirements.txt

3. **Set Environment Variable in heroku/local**
   For local create a .env file and add OPENAI_API_KEY=your_openai_api_key

4. **Run Server**
   flask run
   
## Deploying on Heroku
To deploy your own instance of this API on Heroku, follow these instructions:

## Install the Heroku CLI and log in to your Heroku account.
Create a new app on Heroku: heroku create

## Set the OpenAI API key as a Heroku config var:
heroku config:set OPENAI_API_KEY=your_openai_api_key

## Deploy the application using Git:
git push heroku master


## Front-end Repository
The front-end React application that interacts with this API is hosted separately. You can find the repository here: [AI Interview UI](https://github.com/mohammed-saalim/orahacks-chat-interview-ui)

## Contributions
We welcome contributions to this project. If you have suggestions or improvements, please fork the repository and submit a pull request.

## Showcasing
The project was showcased at OraHacks "Back to the Future" event. Check out the LinkedIn post about this event here. [here](https://www.linkedin.com/posts/orahacks_orahacks-back-to-the-future-was-an-action-ugcPost-7188349494508371968-F9QR?utm_source=share&utm_medium=member_desktop)


## Screenshots


![WhatsApp Image 2024-05-15 at 12 20 48 AM](https://github.com/mohammed-saalim/orahacks-chat-interview-ui/assets/53993039/0898629d-c611-46a4-91ce-c8791a4ee173)



<img width="1518" alt="Screenshot 2024-05-14 at 10 05 17 PM" src="https://github.com/mohammed-saalim/orahacks-chat-interview-ui/assets/53993039/614d8565-dfe3-4358-a6d3-479694f791cd">

<img width="1013" alt="Screenshot 2024-05-14 at 10 05 50 PM" src="https://github.com/mohammed-saalim/orahacks-chat-interview-ui/assets/53993039/488ab751-fdd2-476c-8523-8c02fd5c4818">


<img width="1778" alt="Screenshot 2024-05-14 at 11 53 42 PM" src="https://github.com/mohammed-saalim/orahacks-chat-interview-ui/assets/53993039/4cfa8699-3166-48ca-bd4d-1a1aacd4a314">


<img width="1550" alt="Screenshot 2024-05-14 at 11 54 08 PM" src="https://github.com/mohammed-saalim/orahacks-chat-interview-ui/assets/53993039/744f236a-f948-4aa9-a034-1f26bcdd82c6">






