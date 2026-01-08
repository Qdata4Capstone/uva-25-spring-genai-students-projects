// src/App.js

import React from 'react';
import {
  HashRouter as Router,
  Routes,
  Route,
  Navigate
} from 'react-router-dom';

import { AuthProvider, AuthContext } from './components/AuthContext';
import LoginPage        from './components/LoginPage';
import ForumPage        from './components/ForumPage';
import CreatePostPage   from './components/CreatePostPage';
import CreateAIPostPage from './components/CreateAIPostPage';
import PostDetailPage   from './components/PostDetailPage';

function PrivateRoute({ children }) {
  const { user } = React.useContext(AuthContext);
  return user ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Navigate to="/login" replace />} />
          <Route path="/login" element={<LoginPage />} />

          <Route
            path="/forum"
            element={
              <PrivateRoute>
                <ForumPage />
              </PrivateRoute>
            }
          />

          <Route
            path="/post/:id"
            element={
              <PrivateRoute>
                <PostDetailPage />
              </PrivateRoute>
            }
          />

          <Route
            path="/create"
            element={
              <PrivateRoute>
                <CreatePostPage />
              </PrivateRoute>
            }
          />

          <Route
            path="/create/ai"
            element={
              <PrivateRoute>
                <CreateAIPostPage />
              </PrivateRoute>
            }
          />

          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}



















/*import React from 'react';
import BlogGenerator from './components/BlogGenerator';

function App() {
  return (
    <div style={{ background: '#f0f2f5', minHeight: '100vh', padding: '40px 0' }}>
      <BlogGenerator />
    </div>
  );
}

export default App; */