import Image from "next/image";
import { Geist, Geist_Mono } from "next/font/google";
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

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
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
      const response = await fetch('http://localhost:8000/api/documents/upload/', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        alert('File uploaded successfully!');
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
            <div className="text-sm text-gray-500">
              Supported formats: PDF, TXT, DOC
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
