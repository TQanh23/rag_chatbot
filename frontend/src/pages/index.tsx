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
  const [darkMode, setDarkMode] = useState(false);

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
      const response = await fetch('/api/proxy/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question.trim(),
          filenames: uploadedFiles.map(file => file.name),
        }),
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
    
    try {
      // Read file as ArrayBuffer for consistent handling
      const fileBuffer = await file.arrayBuffer();
      const blob = new Blob([fileBuffer], { type: file.type });
      formData.append('file', blob, file.name);

      const response = await fetch('/api/proxy/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        // Add file to uploaded files list
        setUploadedFiles(prev => [{
          name: file.name,
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
    <div className={`${geistSans.className} min-h-screen flex flex-col transition-colors ${darkMode ? "bg-gray-900 text-gray-100" : "bg-white text-gray-900"}`}>
      {/* Header */}
      <header className={`p-4 flex justify-between items-center shadow ${darkMode ? "bg-gray-800 text-white" : "bg-[#2563EB] text-white"}`}>
        <h1 className="text-2xl font-bold">Rag Chatbot</h1>
        <div className="flex gap-2">
          {/* Toggle Dark/Light Mode */}
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="px-4 py-2 rounded-lg bg-white text-[#2563EB] hover:bg-gray-100 transition-colors font-medium dark:bg-gray-700 dark:text-gray-100 dark:hover:bg-gray-600"
          >
            {darkMode ? "Light Mode" : "Dark Mode"}
          </button>
          <button
            onClick={handleNewChat}
            className="px-4 py-2 bg-white text-[#2563EB] rounded-lg hover:bg-gray-100 transition-colors font-medium dark:bg-gray-700 dark:text-gray-100 dark:hover:bg-gray-600"
          >
            New Chat
          </button>
        </div>
      </header>

      <div className="flex flex-1 p-4 gap-4">
        <div className="flex flex-col flex-1 gap-4">
          {/* Chat Window */}
          <div className={`flex-1 border rounded-lg p-4 min-h-[400px] overflow-y-auto shadow-sm transition-colors ${darkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}>
            {messages.length === 0 ? (
              <div className="text-center text-gray-500 dark:text-gray-400">
                Start a new conversation
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] p-3 rounded-lg shadow-sm ${
                        message.role === 'user'
                          ? "bg-[#DBEAFE] text-[#1E40AF] font-medium dark:bg-blue-600 dark:text-white"
                          : "bg-[#F3F4F6] text-[#111827] dark:bg-gray-700 dark:text-gray-100"
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
              className={`flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-[#2563EB] transition-colors ${
                darkMode
                  ? "bg-gray-800 text-white border-gray-600 placeholder-gray-400"
                  : "bg-white text-[#111827] border-gray-300 placeholder-gray-400"
              }`}
              disabled={loading}
            />
            <button
              type="submit"
              className={`px-4 py-2 rounded-lg transition-colors font-medium ${
                darkMode
                  ? "bg-blue-600 text-white hover:bg-blue-700"
                  : "bg-[#2563EB] text-white hover:bg-blue-700"
              } disabled:opacity-50 disabled:cursor-not-allowed`}
              disabled={loading || !question.trim()}
            >
              {loading ? "Sending..." : "Send"}
            </button>
          </form>
        </div>

        {/* Document Upload Section */}
        <div className={`w-64 border rounded-lg p-4 shadow-sm transition-colors ${darkMode ? "bg-gray-800 border-gray-700" : "bg-[#F9FAFB] border-gray-200"}`}>
          <h2 className="text-lg font-semibold mb-4">Document Upload</h2>
          <div className="flex flex-col gap-4">
            <label className="px-4 py-2 bg-[#2563EB] text-white rounded-lg hover:bg-blue-700 cursor-pointer text-center transition-colors font-medium">
              <input 
                type="file" 
                onChange={handleFileUpload}
                className="hidden"
                accept=".pdf,.txt,.doc,.docx"
              />
              Upload File
            </label>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Supported formats: PDF
            </div>

            {/* Thêm phần hiển thị uploaded files */}
            {uploadedFiles.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-semibold mb-2">Uploaded Documents:</h3>
                <div className="space-y-2">
                  {uploadedFiles.map((file, index) => (
                    <div 
                      key={index}
                      className={`p-2 rounded-lg text-sm flex items-center justify-between ${
                        darkMode
                          ? "bg-gray-700 text-gray-200"
                          : "bg-white text-gray-700 border border-gray-200"
                      }`}
                    >
                      <span className="truncate">{file.name}</span>
                      <button
                        onClick={() => {
                          setUploadedFiles(files => files.filter((_, i) => i !== index));
                        }}
                        className="ml-2 text-red-500 hover:text-red-700"
                      >
                        ×
                      </button>
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