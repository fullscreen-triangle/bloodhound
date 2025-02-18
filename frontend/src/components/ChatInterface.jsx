import React, { useState, useRef, useEffect } from 'react';
import { api } from '../services/api';

export const ChatInterface = ({ experimentId }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    try {
      const response = await api.sendChatMessage({
        message: inputValue,
        experimentId,
        context: {
          previousMessages: messages.slice(-5)
        }
      });
      
      const aiMessage = {
        id: Date.now() + 1,
        text: response.message,
        sender: 'ai',
        timestamp: new Date().toISOString(),
        metadata: response.metadata
      };
      
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: 'Sorry, there was an error processing your message.',
        sender: 'system',
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="chat-interface">
      <div className="chat-messages">
        {messages.map(message => (
          <div 
            key={message.id} 
            className={`message ${message.sender}`}
          >
            <div className="message-content">
              {message.text}
              {message.metadata && (
                <div className="message-metadata">
                  {message.metadata.type === 'plot' && (
                    <ExperimentPlot plotData={message.metadata.plotData} />
                  )}
                  {message.metadata.type === 'graph' && (
                    <KnowledgeGraph data={message.metadata.graphData} />
                  )}
                </div>
              )}
            </div>
            <div className="message-timestamp">
              {new Date(message.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask about your experiment..."
          disabled={isLoading}
          className="chat-input"
        />
        <button 
          type="submit" 
          disabled={isLoading}
          className="chat-submit"
        >
          {isLoading ? 'Thinking...' : 'Send'}
        </button>
      </form>
    </div>
  );
}; 