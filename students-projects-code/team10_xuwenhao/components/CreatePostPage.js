// src/components/CreatePostPage.jsx

import React, { useContext, useState, useEffect } from 'react';
import MarkdownIt from 'markdown-it';
import MdEditor from 'react-markdown-editor-lite';
import 'react-markdown-editor-lite/lib/index.css';
import { AuthContext } from './AuthContext';
import { useNavigate, useLocation } from 'react-router-dom';

const mdParser = new MarkdownIt();

export default function CreatePostPage() {
  const { user } = useContext(AuthContext);
  const navigate = useNavigate();
  const location = useLocation();

  const editIndex = location.state?.editIndex;
  const editPost  = location.state?.post;
  const aiContent = location.state?.aiContent;

  const [title, setTitle]     = useState(editPost?.title || '');
  const [content, setContent] = useState(editPost?.content || '');

  //Permission verification
  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }
    if (editPost && editPost.author !== user.username) {
      alert('✋ You do not have permission to edit this blog');
      navigate('/forum');
      return;
    }
  }, [user, editPost, navigate]);

  // AI assistance
  useEffect(() => {
    if (aiContent) {
      setContent(aiContent);
    }
  }, [aiContent]);

  const onSubmit = e => {
    e.preventDefault();
    const all = JSON.parse(localStorage.getItem('posts') || '[]');
    const newPost = {
      title:    title || content.slice(0, 50) + '…',
      author:   user.username,
      date:     editPost?.date || new Date().toISOString(),
      content,
      likes:    editPost?.likes || 0,
      comments: editPost?.comments || []
    };

    if (editIndex != null) {
      all[editIndex] = newPost;
    } else {
      all.unshift(newPost);
    }
    localStorage.setItem('posts', JSON.stringify(all));
    navigate('/forum');
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
      <button onClick={() => navigate('/forum')} style={{ marginBottom: 12 }}>
        &larr; Back to Forum
      </button>

      <h2>{editPost ? 'Edit Blog' : 'Write a new blog'}</h2>

      <button
        onClick={() =>
          navigate('/create/ai', { state: { editIndex, post: editPost } })
        }
        style={{
          margin: '12px 0',
          padding: '8px 16px',
          background: '#17a2b8',
          border: 'none',
          color: '#fff',
          borderRadius: 4,
          cursor: 'pointer'
        }}
      >
        AI Assistance
      </button>

      <form onSubmit={onSubmit}>
        <input
          type="text"
          placeholder="Title"
          value={title}
          onChange={e => setTitle(e.target.value)}
          style={{
            width: '100%',
            padding: 8,
            fontSize: 16,
            marginBottom: 12,
            border: '1px solid #ccc',
            borderRadius: 4
          }}
        />

        <MdEditor
          value={content}
          style={{ height: '600px', marginBottom: '12px' }}
          renderHTML={text => mdParser.render(text)}
          onChange={({ text }) => setContent(text)}
        />

        <button
          type="submit"
          style={{
            padding: '12px 24px',
            background: '#28a745',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer'
          }}
        >
          {editPost ? 'Save' : 'Post'}
        </button>
      </form>
    </div>
  );
}
