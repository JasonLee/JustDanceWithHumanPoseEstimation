import React, { useState } from 'react';
// import './App.css';
import SongList from './components/SongList';
import Login from './components/Login';
import useToken from './hooks/useToken';
import Register from './components/Register';
import SongGame from './components/SongGame';
import SongEndScore from './components/SongEndScore';
import Multiplayer from './components/Multiplayer';
import { BrowserRouter, Route, Switch, Redirect } from 'react-router-dom';

function App() {
	const { token, setToken } = useToken();

	return (
		<div className="App">
			<header className="App-header">
				<BrowserRouter>
					<Switch>
						<Route path="/songs/:lobbyID?">
							<SongList token={token}/>
						</Route>
						<Route path="/login">
							<Login setToken={setToken} />
						</Route>
						<Route path="/register">
							<Register />
						</Route>
						<Route path="/game/:songID/:lobbyID?">
							<SongGame />
						</Route>
						<Route path="/results/:gameID/:lobbyID?">
							<SongEndScore />
						</Route>
						<Route path="/multiplayer/:lobbyID/:songID?">
							<Multiplayer />
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