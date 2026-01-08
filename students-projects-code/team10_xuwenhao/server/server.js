// server.js
import express from 'express';
import multer from 'multer';
import pdf from 'pdf-parse';

const app = express();
const upload = multer();

app.post('/api/extract', upload.single('file'), async (req, res) => {
  try {
    const data = await pdf(req.file.buffer);
    res.json({ text: data.text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'PDF Parsing failed' });
  }
});

app.listen(3001, () => console.log('PDF‐parse Service started'));
