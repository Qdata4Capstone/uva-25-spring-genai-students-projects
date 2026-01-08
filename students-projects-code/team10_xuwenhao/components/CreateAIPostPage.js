import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import BlogGenerator from './BlogGenerator';

export default function CreateAIPostPage() {
  const navigate = useNavigate();
  const { state } = useLocation();
  const editIndex = state?.editIndex;
  const editPost = state?.post;

  const handleComplete = generated => {
    
    navigate('/create', {
      state: {
        editIndex,
        post: editPost,
        aiContent: generated
      }
    });
  };

  const wrapper = {
    width: '100%',
    maxWidth: 1400,
    margin: '0 auto',
    padding: '24px 16px',
    fontFamily: 'Arial, sans-serif'
  };

  return (
    <div style={wrapper}>
      <button
        onClick={() =>
          navigate('/create', { state: { editIndex, post: editPost } })
        }
        style={{ marginBottom: 12 }}
      >
        &larr; Return to Write
      </button>
      <BlogGenerator onComplete={handleComplete} />
    </div>
  );
}
