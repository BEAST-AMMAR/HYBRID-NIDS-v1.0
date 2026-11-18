import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate, Link, useLocation } from 'react-router-dom';

const Login = ({ setToken }) => {
    const location = useLocation();
    const [email, setEmail] = useState('admin');
    const [password, setPassword] = useState('admin');
    const [error, setError] = useState('');
    const navigate = useNavigate();

    useEffect(() => {
        // Check for token in URL (from OAuth redirect)
        const params = new URLSearchParams(location.search);
        const token = params.get('token');
        if (token) {
            setToken(token);
            localStorage.setItem('token', token);
            navigate('/monitor');
        }
    }, [location, setToken, navigate]);

    const handleGoogleLogin = () => {
        window.location.href = 'http://localhost:8000/login/google';
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        const params = new URLSearchParams();
        params.append('username', email);
        params.append('password', password);

        const host = window.location.hostname || '127.0.0.1';
        try {
            const response = await axios.post(`http://${host}:8000/token`, params);
            setToken(response.data.access_token);
            localStorage.setItem('token', response.data.access_token);
            navigate('/monitor');
        } catch (err) {
            setError('Invalid email or password');
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-900">
            <div className="bg-gray-800 p-8 rounded-lg shadow-xl w-96">
                <h2 className="text-3xl font-bold text-white mb-4 text-center">NIDS Login</h2>
                
                <div className="mb-5 p-3 bg-blue-950/60 border border-blue-500/40 rounded-lg text-xs text-blue-200 flex justify-between items-center shadow-inner">
                    <div>
                        <p className="font-semibold text-blue-300">Default Admin Credentials</p>
                        <p className="text-[11px] text-gray-400 mt-0.5">User: <code className="text-white bg-blue-900/60 px-1 rounded">admin</code> | Pass: <code className="text-white bg-blue-900/60 px-1 rounded">admin</code></p>
                    </div>
                    <button 
                        type="button" 
                        onClick={() => { setEmail('admin'); setPassword('admin'); }}
                        className="text-xs bg-blue-600 hover:bg-blue-500 text-white font-medium px-2.5 py-1 rounded transition shadow-sm"
                    >
                        Auto-Fill
                    </button>
                </div>

                {error && <p className="text-red-500 mb-4 text-center">{error}</p>}
                <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                        <label className="block text-gray-400 mb-2">Email or Username</label>
                        <input
                            type="text"
                            className="w-full p-2 rounded bg-gray-700 text-white border border-gray-600 focus:outline-none focus:border-blue-500"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>
                    <div className="mb-6">
                        <label className="block text-gray-400 mb-2">Password</label>
                        <input
                            type="password"
                            className="w-full p-2 rounded bg-gray-700 text-white border border-gray-600 focus:outline-none focus:border-blue-500"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>
                    <button
                        type="submit"
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 rounded transition duration-200"
                    >
                        Login
                    </button>
                </form>
                <p className="text-gray-400 mt-4 text-center">
                    Don't have an account? <Link to="/signup" className="text-blue-500 hover:underline">Sign up</Link>
                </p>
                <div className="mt-6 flex flex-col space-y-2">
                    <button 
                        onClick={handleGoogleLogin}
                        className="bg-white text-gray-700 py-2 rounded flex items-center justify-center font-semibold hover:bg-gray-100 transition"
                    >
                        Sign in with Google
                    </button>
                    <button className="bg-gray-700 text-white py-2 rounded flex items-center justify-center font-semibold border border-gray-600">
                        Sign in with GitHub
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Login;
