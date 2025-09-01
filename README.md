# RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents and ask document-related questions. The chatbot leverages the Next.js framework for the frontend and Django for the backend.

## Features

- **Document Upload**: Users can upload documents to the system.
- **Question Answering**: The chatbot answers questions based on the uploaded documents.
- **Modern Frontend**: Built with Next.js for a fast and responsive user interface.
- **Robust Backend**: Powered by Django to handle API requests and manage data.

## Tech Stack

- **Frontend**: Next.js (TypeScript)
- **Backend**: Django (Python)
- **Database**: SQLite (default, can be replaced with other databases)

## Project Structure

### Backend

The backend is located in the `backend/` directory and is a Django project. Key files and directories include:

- `manage.py`: Django's command-line utility.
- `api/`: Contains the Django app for handling API requests.
  - `models.py`: Defines the database models.
  - `views.py`: Contains the logic for handling requests.
  - `urls.py`: Maps URLs to views.
- `settings.py`: Django project settings.

### Frontend

The frontend is located in the `frontend/` directory and is a Next.js project. Key files and directories include:

- `src/pages/`: Contains the Next.js pages.
  - `index.tsx`: The main landing page.
  - `api/hello.ts`: Example API route.
- `public/`: Static assets like images and icons.
- `styles/globals.css`: Global CSS styles.

## Getting Started

### Prerequisites

- Node.js and npm installed for the frontend.
- Python and pip installed for the backend.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TQanh23/rag_chatbot.git
   cd rag_chatbot
   ```

2. Set up the backend:
   ```bash
   cd backend
   python -m venv .venv
   .\.venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   python manage.py migrate
   python manage.py runserver
   ```

3. Set up the frontend:
   ```bash
   cd ../frontend
   npm install
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:3000` to access the application.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.


## Acknowledgments

- Inspired by modern chatbot architectures and RAG techniques.
- Built with love using Next.js and Django.
