import React, { useState } from 'react';
import './App.css';
import SongList from './components/SongList';
import Login from './components/Login';
import useToken from './hooks/useToken';
import Register from './components/Register';
import SongGame from './components/SongGame';
import { BrowserRouter, Route, Switch, Redirect } from 'react-router-dom';

function App() {
	const { token, setToken } = useToken();

	return (
		<div className="App">
			<header className="App-header">
				<BrowserRouter>
					<Switch>
						<Route path="/songs">
							<SongList />
						</Route>
						<Route path="/login">
							<Login setToken={setToken} />
						</Route>
						<Route path="/register">
							<Register />
						</Route>
						<Route path="/test">
							<SongGame />
						</Route>
						<Route path="/" render={() => {
							return (
								token ?
								<Redirect to="/songs" /> :
								<Redirect to="/login" /> 
							)
						}}>
						</Route>
					</Switch>
				</BrowserRouter>
			</header>
		</div>
	);
}

export default App;