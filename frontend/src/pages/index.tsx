import Image from "next/image";
import { Geist, Geist_Mono } from "next/font/google";
import { useEffect } from 'react';
import { useState } from "react";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface UploadedFile {
  name: string;
  content?: string;
  uploadTime: Date;
}

export default function Home() {
  useEffect(() => {
    // Add Tahoma font to the document
    document.body.style.fontFamily = 'Tahoma, sans-serif';
  }, []);

  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [dbAction, setDbAction] = useState<'init' | 'delete' | null>(null);

  const handleNewChat = () => {
    setMessages([]);
    setQuestion('');
  };

  const handleQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    try {
      setLoading(true);
      // Add user message
      const userMessage: Message = { role: 'user', content: question };
      setMessages(prev => [...prev, userMessage]);
      
      // Send question to API
      const response = await fetch('http://localhost:8000/api/ask/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question.trim() }),
      });

      if (!response.ok) throw new Error('Failed to get answer');
      
      const data = await response.json();
      
      // Add assistant message
      const assistantMessage: Message = { role: 'assistant', content: data.answer };
      setMessages(prev => [...prev, assistantMessage]);
      
      // Clear input
      setQuestion('');
    } catch (error) {
      console.error('Error getting answer:', error);
      alert('Failed to get answer');
    } finally {
      setLoading(false);
    }
  };

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Read file content if it's a text file
      let fileContent: string | undefined;
      if (file.type === 'text/plain') {
        fileContent = await file.text();
      }

      const response = await fetch('http://localhost:8000/api/upload/', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        // Add file to uploaded files list
        setUploadedFiles(prev => [{
          name: file.name,
          content: fileContent,
          uploadTime: new Date()
        }, ...prev]);
      } else {
        alert('Failed to upload file');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file');
    }
  };

  return (
    <div className={`${geistSans.className} ${geistMono.className} min-h-screen flex flex-col`}>
      {/* Header */}
      <header className="bg-red-600 text-white p-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold">Rag Chatbot</h1>
        <button
          onClick={handleNewChat}
          className="px-4 py-2 bg-white text-red-600 rounded-lg hover:bg-gray-100"
        >
          New Chat
        </button>
      </header>

      <div className="flex flex-1 p-4 gap-4">
        <div className="flex flex-col flex-1 gap-4">
          {/* Chat Window */}
          <div className="flex-1 border rounded-lg p-4 bg-gray-50 min-h-[400px] overflow-y-auto">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500">
                Start a new conversation
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${
                      message.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    <div
                      className={`max-w-[80%] p-3 rounded-lg ${
                        message.role === 'user'
                          ? 'bg-red-600 text-white'
                          : 'bg-gray-200 text-gray-800'
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Question Input */}
          <form onSubmit={handleQuestion} className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Type your question here..."
              className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
              disabled={loading}
            />
            <button
              type="submit"
              className={`px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed`}
              disabled={loading || !question.trim()}
            >
              {loading ? 'Sending...' : 'Send'}
            </button>
          </form>
        </div>

        {/* Document Upload Section */}
        <div className="w-64 border rounded-lg p-4 bg-gray-50">
          <h2 className="text-lg font-semibold mb-4">Document Upload</h2>
          <div className="flex flex-col gap-4">
            <label className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 cursor-pointer text-center">
              <input 
                type="file" 
                onChange={handleFileUpload}
                className="hidden"
                accept=".pdf,.txt,.doc,.docx"
              />
              Upload File
            </label>
            <div className="text-sm text-gray-500 mb-4">
              Supported formats: PDF, TXT, DOC
            </div>

            {/* Uploaded Files List */}
            {uploadedFiles.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-semibold mb-2">Uploaded Files</h3>
                <div className="max-h-60 overflow-y-auto">
                  {uploadedFiles.map((file, index) => (
                    <div key={index} className="mb-4 p-2 border rounded bg-white">
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          <Image
                            src="/file.svg"
                            alt="File icon"
                            width={16}
                            height={16}
                          />
                          <span className="text-sm font-medium truncate">
                            {file.name}
                          </span>
                        </div>
                        <button
                          onClick={() => {
                            if (confirm('Are you sure you want to remove this file from the list?')) {
                              setUploadedFiles(prev => prev.filter((_, i) => i !== index));
                            }
                          }}
                          className="p-1 hover:bg-red-100 rounded-full"
                          title="Remove from list"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-red-600" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                          </svg>
                        </button>
                      </div>
                      <div className="text-xs text-gray-500 mb-1">
                        Uploaded: {file.uploadTime.toLocaleString()}
                      </div>
                      {file.content && (
                        <div className="mt-2 p-2 bg-gray-50 rounded text-xs max-h-32 overflow-y-auto">
                          <pre className="whitespace-pre-wrap">{file.content}</pre>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}