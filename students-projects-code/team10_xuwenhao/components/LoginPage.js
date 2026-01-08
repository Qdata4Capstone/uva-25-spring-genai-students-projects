// src/components/LoginPage.jsx

import React, { useState, useContext } from 'react';
import { AuthContext } from './AuthContext';
import { useNavigate } from 'react-router-dom';

export default function LoginPage() {
  const { login, signup } = useContext(AuthContext);
  const [mode, setMode] = useState('login'); // 'login' or 'signup'
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const nav = useNavigate();

  const onSubmit = async e => {
    e.preventDefault();
    try {
      setError('');
      if (mode === 'login') await login(username, password);
      else await signup(username, password);
      nav('/forum');
    } catch (err) {
      setError(err.message);
    }
  };

  const container = {
    width: '360px',
    margin: '100px auto',
    padding: '24px',
    background: '#fff',
    borderRadius: 8,
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    fontFamily: 'Arial, sans-serif'
  };
  const input = {
    width: '100%',
    padding: 8,
    marginBottom: 12,
    border: '1px solid #ccc',
    borderRadius: 4
  };
  const btn = {
    ...input,
    cursor: 'pointer',
    background: '#007bff',
    color: '#fff',
    border: 'none'
  };

  return (
    <div style={container}>
      <h2 style={{ textAlign: 'center' }}>
        {mode === 'login' ? 'Log in' : 'Create Account'}
      </h2>
      {error && <div style={{ color: 'red', marginBottom: 12 }}>{error}</div>}
      <form onSubmit={onSubmit}>
        <input
          style={input}
          placeholder="Username"
          value={username}
          onChange={e => setUsername(e.target.value)}
        />
        <input
          style={input}
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
        />
        <button style={btn}>{mode === 'login' ? 'Log in' : 'Register'}</button>
      </form>
      <p style={{ textAlign: 'center', marginTop: 12 }}>
        {mode === 'login' ? 'No account?' : 'Already have an account?'}{' '}
        <span
          onClick={() => { setMode(mode === 'login' ? 'signup' : 'login'); setError(''); }}
          style={{ color: '#007bff', cursor: 'pointer' }}
        >
          {mode === 'login' ? 'Create one' : 'Go to login'}
        </span>
      </p>
    </div>
  );
}
