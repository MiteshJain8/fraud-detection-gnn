import { useState } from 'react';
import useAuth from './useAuth';
import axios from '../api/axiosInstance';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const [form, setForm] = useState({ username: '', password: '' });
  const { login } = useAuth();
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    const res = await axios.post('/login', form);
    login(res.data.access_token);
    navigate('/dashboard');
  }

  return (
    <div className="flex flex-col items-center p-8">
      <h1 className="text-2xl font-bold mb-4">Login</h1>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-64">
        <input type="text" placeholder="Username" className="p-2 border" onChange={e => setForm({ ...form, username: e.target.value })} />
        <input type="password" placeholder="Password" className="p-2 border" onChange={e => setForm({ ...form, password: e.target.value })} />
        <button className="bg-blue-600 text-white py-2 rounded" type="submit">Login</button>
      </form>
    </div>
  );
}