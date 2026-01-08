// src/components/PostDetailPage.jsx

import React, { useState, useEffect, useContext } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useParams, useNavigate } from 'react-router-dom';
import { AuthContext } from './AuthContext';

export default function PostDetailPage() {
  const { user } = useContext(AuthContext);
  const { id } = useParams();
  const navigate = useNavigate();

  const [post, setPost] = useState(null);
  const [commentText, setCommentText] = useState('');

  const wrapper = {
    width: '100%',
    maxWidth: 1400,
    margin: '0 auto',
    padding: '24px 16px',
    fontFamily: 'Arial, sans-serif'
  };
  const btn = {
    padding: '8px 16px',
    marginRight: 8,
    cursor: 'pointer',
    border: 'none',
    borderRadius: 4
  };

  useEffect(() => {
    const all = JSON.parse(localStorage.getItem('posts') || '[]');
    const idx = parseInt(id, 10);
    if (!isNaN(idx) && all[idx]) {
      setPost({ ...all[idx], idx });
    }
  }, [id]);

  if (!post) {
    return <div style={wrapper}>The post was not found.</div>;
  }

  // Like
  const handleLike = () => {
    const updated = { ...post, likes: (post.likes || 0) + 1 };
    setPost(updated);
    const all = JSON.parse(localStorage.getItem('posts') || '[]');
    all[post.idx] = updated;
    localStorage.setItem('posts', JSON.stringify(all));
  };

  // Comment
  const handleComment = e => {
    e.preventDefault();
    if (!commentText.trim()) return;
    const updated = {
      ...post,
      comments: [...(post.comments || []), { text: commentText, date: Date.now(), author: user.username }]
    };
    setPost(updated);
    const all = JSON.parse(localStorage.getItem('posts') || '[]');
    all[post.idx] = updated;
    localStorage.setItem('posts', JSON.stringify(all));
    setCommentText('');
  };

  return (
    <div style={wrapper}>
      <button onClick={() => navigate('/forum')} style={{ marginBottom: 12 }}>&larr; Back to Forum</button>
      <h1>{post.title}</h1>
      <p style={{ color: '#666', fontSize: 14 }}>
        by {post.author} • {new Date(post.date).toLocaleString()}
      </p>

      {/* Markdown */}
      <div style={{ margin: '24px 0' }} className="markdown-body">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {post.content}
        </ReactMarkdown>
      </div>

      {/* Like */}
      <button onClick={handleLike} style={{ ...btn, background: '#ffc107' }}>
        👍 {post.likes || 0}
      </button>

      {/* Comment */}
      <h3 style={{ marginTop: 32 }}>Comments ({(post.comments || []).length})</h3>
      {(post.comments || []).map((c, i) => (
        <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #eee' }}>
          <p style={{ margin: 0, fontSize: 14, color: '#333' }}>{c.text}</p>
          <p style={{ margin: '4px 0 0 0', fontSize: 12, color: '#999' }}>
            by {c.author} • {new Date(c.date).toLocaleString()}
          </p>
        </div>
      ))}

      {/* Add comment */}
      {user && (
        <form onSubmit={handleComment} style={{ marginTop: 24 }}>
          <textarea
            value={commentText}
            onChange={e => setCommentText(e.target.value)}
            placeholder="Write your comment..."
            style={{
              width: '100%',
              padding: 8,
              border: '1px solid #ccc',
              borderRadius: 4,
              resize: 'vertical'
            }}
          />
          <button type="submit" style={{ ...btn, background: '#007bff', color: '#fff', marginTop: 8 }}>
            Post
          </button>
        </form>
      )}
    </div>
  );
}
