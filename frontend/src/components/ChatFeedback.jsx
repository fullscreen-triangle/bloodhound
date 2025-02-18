import React, { useState } from 'react';
import { api } from '../services/api';

export const ChatFeedback = ({ message, onFeedbackSubmit }) => {
  const [rating, setRating] = useState(null);
  const [comments, setComments] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!rating) return;
    
    try {
      setIsSubmitting(true);
      
      await api.sendChatFeedback({
        messageId: message.id,
        query: message.query,
        response: message.text,
        context: message.context,
        rating,
        comments
      });
      
      if (onFeedbackSubmit) {
        onFeedbackSubmit({ rating, comments });
      }
      
    } catch (error) {
      console.error('Feedback error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <form onSubmit={handleSubmit} className="feedback-form">
      <div className="rating-buttons">
        {[1, 2, 3, 4, 5].map((value) => (
          <button
            key={value}
            type="button"
            className={`rating-button ${rating === value ? 'selected' : ''}`}
            onClick={() => setRating(value)}
          >
            {value}
          </button>
        ))}
      </div>
      
      <textarea
        value={comments}
        onChange={(e) => setComments(e.target.value)}
        placeholder="Additional comments (optional)"
        className="feedback-comments"
      />
      
      <button 
        type="submit"
        disabled={!rating || isSubmitting}
        className="feedback-submit"
      >
        {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
      </button>
    </form>
  );
}; 