import React, { useState } from 'react';
import './App.css';
import SongList from './components/SongList';
import Login from './components/Login';
import ProfileBox from './components/ProfileBox';
import useToken from './hooks/useToken';
import Register from './components/Register';

function App() {
	const { token, setToken } = useToken();

	if (!token) {
		return <Login setToken={setToken} />
	}

	return (
		<div className="App">
			<header className="App-header">
				<ProfileBox />
				<SongList />
			</header>
		</div>
	);
}

export default App;