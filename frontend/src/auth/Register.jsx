import { useState } from 'react';
import axios from '../api/axiosInstance';
import { useNavigate } from 'react-router-dom';

export default function Register() {
  const [form, setForm] = useState({ username: '', password: '' });
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    await axios.post('/register', form);
    navigate('/login');
  }

  return (
    <div className="flex flex-col items-center p-8">
      <h1 className="text-2xl font-bold mb-4">Register</h1>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-64">
        <input type="text" placeholder="Username" className="p-2 border" onChange={e => setForm({ ...form, username: e.target.value })} />
        <input type="password" placeholder="Password" className="p-2 border" onChange={e => setForm({ ...form, password: e.target.value })} />
        <button className="bg-green-600 text-white py-2 rounded" type="submit">Register</button>
      </form>
    </div>
  );
}