// src/components/BlogGenerator.jsx

import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { generateBlogPost } from '../api/openai';
import { BASE_SYSTEM_PROMPT } from '../prompts/systemPrompt';

export default function BlogGenerator({ onComplete }) {
  // 动态从 CDN 加载 PDF.js
  useEffect(() => {
    if (!window.pdfjsLib) {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.min.js';
      script.onload = () => {
        window.pdfjsLib.GlobalWorkerOptions.workerSrc =
          'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js';
      };
      document.body.appendChild(script);
      return () => document.body.removeChild(script);
    }
  }, []);

  const [topic, setTopic] = useState('');
  const [keywords, setKeywords] = useState([]);
  const [kwInput, setKwInput] = useState('');
  const [background, setBackground] = useState('');
  const [writingStyle, setWritingStyle] = useState('');
  const [wordCount, setWordCount] = useState(3000);
  const [extraPrompt, setExtraPrompt] = useState('');

  const [uploadedFile, setUploadedFile] = useState(null);
  const [pdfText, setPdfText] = useState('');
  const [generatedPost, setGeneratedPost] = useState('');
  const [loading, setLoading] = useState(false);

  const fileRef = useRef();

  // 回车添加关键词
  const onKwKeyDown = e => {
    if (e.key === 'Enter' && kwInput.trim()) {
      e.preventDefault();
      setKeywords([...keywords, kwInput.trim()]);
      setKwInput('');
    }
  };

  // 删除关键词
  const removeKeyword = idx => {
    setKeywords(keywords.filter((_, i) => i !== idx));
  };

  // 读取 PDF 并提取文字
  const onFileChange = async e => {
    const file = e.target.files[0];
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please upload a PDF');
      fileRef.current.value = '';
      setUploadedFile(null);
      setPdfText('');
      return;
    }
    setUploadedFile(file);

    // 等待 pdf.js 加载完成
    if (!window.pdfjsLib) {
      alert('PDF.js is still loading, please try again in a moment.');
      return;
    }

    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = window.pdfjsLib.getDocument({ data: arrayBuffer });
    try {
      const pdf = await loadingTask.promise;
      let allText = '';
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const { items } = await page.getTextContent();
        allText += items.map(item => item.str).join(' ') + '\n\n';
      }
      setPdfText(allText);
    } catch (err) {
      console.error('PDF parse error:', err);
      alert('Failed to parse PDF');
      setPdfText('');
    }
  };

  // 调用 OpenAI 生成博客
  const onGenerate = async () => {
    if (!topic.trim() && !pdfText.trim()) {
      alert('Enter a topic or upload a PDF');
      return;
    }
    const parts = [
      BASE_SYSTEM_PROMPT.trim(),
      topic && `Topic: ${topic}`,
      keywords.length && `Keywords: ${keywords.join(', ')}`,
      background && `Background: ${background}`,
      writingStyle && `Writing Style: ${writingStyle}`,
      `Desired Word Count: ${wordCount}`,
      pdfText && `PDF Excerpt:\n"""${pdfText.slice(0, 3000)}"""`,
      extraPrompt && `Extra Instructions: ${extraPrompt}`,
      'Please write the blog post now.'
    ].filter(Boolean).join('\n\n');

    setLoading(true);
    try {
      const output = await generateBlogPost(parts);
      if (onComplete) {
        onComplete(output);
      } else {
        setGeneratedPost(output);
      }
    } catch (err) {
      console.error(err);
      alert('❌ Generation failed');
    } finally {
      setLoading(false);
    }
  };

  // 样式
  const container = {
    padding: '20px',
    fontFamily: 'Arial, sans-serif',
    background: '#fff',
    borderRadius: 8,
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  };
  const labelRow = { display: 'flex', justifyContent: 'space-between', marginBottom: 6 };
  const inputBase = { width: '100%', padding: 8, fontSize: 14, border: '1px solid #ccc', borderRadius: 4 };
  const textarea = { ...inputBase, resize: 'vertical' };
  const tag = {
    display: 'inline-flex',
    alignItems: 'center',
    background: '#e2e2e2',
    padding: '4px 8px',
    borderRadius: 16,
    marginRight: 6,
    marginBottom: 6,
    fontSize: 13
  };
  const removeBtn = { marginLeft: 6, border: 'none', background: 'transparent', cursor: 'pointer' };
  const generateBtn = {
    width: '100%',
    padding: '12px',
    fontSize: 16,
    background: '#007bff',
    color: '#fff',
    border: 'none',
    borderRadius: 4,
    cursor: 'pointer'
  };

  return (
    <div style={container}>
      <h1 style={{ textAlign: 'center' }}>AI Blog Generator</h1>

      {/* Topic */}
      <div style={{ marginBottom: 20 }}>
        <div style={labelRow}>
          <label htmlFor="topic"><strong>Topic</strong></label>
        </div>
        <input
          id="topic"
          type="text"
          placeholder="Eg: Best AI Tools For Blogging"
          value={topic}
          onChange={e => setTopic(e.target.value)}
          style={inputBase}
        />
      </div>

      {/* Keywords */}
      <div style={{ marginBottom: 20 }}>
        <div style={labelRow}>
          <label><strong>Keywords</strong></label>
          <small style={{ color: '#666' }}>Optional</small>
        </div>
        <div style={{ border: '1px solid #ccc', borderRadius: 4, padding: 8, minHeight: 44 }}>
          {keywords.map((k, i) => (
            <span key={i} style={tag}>
              {k}
              <button type="button" style={removeBtn} onClick={() => removeKeyword(i)}>×</button>
            </span>
          ))}
          <input
            type="text"
            value={kwInput}
            onChange={e => setKwInput(e.target.value)}
            onKeyDown={onKwKeyDown}
            placeholder="Type & Enter"
            style={{ ...inputBase, border: 'none', boxShadow: 'none' }}
          />
        </div>
      </div>

      {/* Background */}
      <div style={{ marginBottom: 20 }}>
        <div style={labelRow}>
          <label htmlFor="bg"><strong>Background</strong></label>
          <small style={{ color: '#666' }}>Optional</small>
        </div>
        <textarea
          id="bg"
          rows={3}
          placeholder="Any facts, stats..."
          value={background}
          onChange={e => setBackground(e.target.value)}
          style={textarea}
        />
      </div>

      {/* Writing Style */}
      <div style={{ marginBottom: 20 }}>
        <div style={labelRow}>
          <label htmlFor="style"><strong>Writing Style</strong></label>
        </div>
        <textarea
          id="style"
          rows={2}
          placeholder="Confident yet humble..."
          value={writingStyle}
          onChange={e => setWritingStyle(e.target.value)}
          style={textarea}
        />
      </div>

      {/* Word Count */}
      <div style={{ marginBottom: 20 }}>
        <div style={labelRow}>
          <label htmlFor="words"><strong>Words</strong></label>
          <small>{wordCount}</small>
        </div>
        <input
          id="words"
          type="range"
          min={500}
          max={5000}
          step={100}
          value={wordCount}
          onChange={e => setWordCount(e.target.value)}
          style={{ width: '100%' }}
        />
      </div>

      {/* PDF Upload */}
      <div style={{ marginBottom: 20 }}>
        <div style={labelRow}>
          <label><strong>Upload PDF</strong></label>
        </div>
        <input ref={fileRef} type="file" accept=".pdf" onChange={onFileChange} />
        {uploadedFile && <div style={{ marginTop: 8 }}><em>{uploadedFile.name}</em></div>}
      </div>

      {/* Extra Instructions */}
      <div style={{ marginBottom: 20 }}>
        <div style={labelRow}>
          <label htmlFor="extra"><strong>Extra Instructions</strong></label>
          <small style={{ color: '#666' }}>Optional</small>
        </div>
        <textarea
          id="extra"
          rows={2}
          placeholder="Any extra guidance?"
          value={extraPrompt}
          onChange={e => setExtraPrompt(e.target.value)}
          style={textarea}
        />
      </div>

      {/* Generate Button */}
      <button onClick={onGenerate} disabled={loading} style={generateBtn}>
        {loading ? 'Generating…' : 'Generate Blog Post'}
      </button>

      {/* Markdown Preview */}
      {generatedPost && !onComplete && (
        <div style={{ marginTop: 30, textAlign: 'left' }}>
          <h2>Generated Blog Post Preview</h2>
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {generatedPost}
            </ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}
