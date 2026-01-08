// src/components/ForumPage.js

import React, { useState, useEffect, useContext } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { AuthContext } from './AuthContext';
import { Link, useNavigate } from 'react-router-dom';

export default function ForumPage() {
  const { user, logout } = useContext(AuthContext);
  const [posts, setPosts] = useState([]);
  const nav = useNavigate();

  useEffect(() => {
    setPosts(JSON.parse(localStorage.getItem('posts') || '[]'));
  }, []);

  // Del
  const handleDelete = idx => {
    if (!window.confirm('Are you sure you want to delete?')) return;
    const updated = [...posts];
    updated.splice(idx, 1);
    setPosts(updated);
    localStorage.setItem('posts', JSON.stringify(updated));
  };

  const btnSmall = {
    padding: '4px 8px',
    border: 'none',
    borderRadius: 4,
    cursor: 'pointer'
  };

  const cardStyle = {
    background: '#fff',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    padding: '16px',
    marginBottom: '16px',
    position: 'relative'
  };

  // In reverse chronological order
  const sortedPosts = [...posts].sort(
    (a, b) => new Date(b.date) - new Date(a.date)
  );

  return (
    <>
      <header
        style={{
          padding: 16,
          borderBottom: '1px solid #ccc',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}
      >
        <h1>Blog Forum</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {user ? (
            <>
              <button
                onClick={() => nav('/create')}
                style={{ ...btnSmall, background: '#28a745', color: '#fff' }}
              >
                New Blog
              </button>
              <span>Welcome, {user.username}</span>
              <button
                onClick={() => {
                  logout()
                  nav('/login')
                }}
                style={{ ...btnSmall, background: '#dc3545', color: '#fff' }}
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => nav('/login')}
                style={{ ...btnSmall, background: '#007bff', color: '#fff' }}
              >
                Login
              </button>
              <button
                onClick={() => nav('/register')}
                style={{ ...btnSmall, background: '#17a2b8', color: '#fff' }}
              >
                Register
              </button>
            </>
          )}
        </div>
      </header>

      <main style={{ padding: 16 }}>
        {sortedPosts.length === 0 && <p>No posts yet</p>}

        {sortedPosts.map((post, i) => (
          <div key={i} style={cardStyle}>
            <Link
              to={`/post/${i}`}
              style={{ textDecoration: 'none', color: 'inherit' }}
            >
              <h2>{post.title}</h2>
              <p style={{ fontSize: 12, color: '#666' }}>
                by {post.author} • {new Date(post.date).toLocaleString()}
              </p>
              <div style={{ color: '#333', fontSize: 14, margin: '8px 0' }}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {post.content.length > 200
                    ? post.content.slice(0, 200) + '…'
                    : post.content}
                </ReactMarkdown>
              </div>
              {post.image && (
                <img
                  src={post.image}
                  alt=""
                  style={{ maxWidth: '100%', margin: '8px 0' }}
                />
              )}
            </Link>

            {/* Edit/Delete */}
            {user && post.author === user.username && (
              <div
                style={{
                  position: 'absolute',
                  top: 16,
                  right: 16,
                  display: 'flex',
                  gap: 8
                }}
              >
                <button
                  style={{ ...btnSmall, background: '#ffc107' }}
                  onClick={e => {
                    e.preventDefault()
                    nav('/create', { state: { editIndex: i, post } })
                  }}
                >
                  Edit
                </button>
                <button
                  style={{ ...btnSmall, background: '#dc3545', color: '#fff' }}
                  onClick={e => {
                    e.preventDefault()
                    handleDelete(i)
                  }}
                >
                  Delete
                </button>
              </div>
            )}
          </div>
        ))}
      </main>
    </>
  );
}
