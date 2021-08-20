import React, { useState } from 'react';
import styles from './App.module.css';
import SongList from './components/SongList';
import Login from './components/Login';
import useToken from './hooks/useToken';
import Register from './components/Register';
import SongGame from './components/SongGame';
import SongEndScore from './components/SongEndScore';
import Multiplayer from './components/Multiplayer';
import PrivateRoute  from './PrivateRoute';
import StartPage  from './components/StartPage';
import EmailConfirm from './components/EmailConfirm';

import { BrowserRouter, Route, Switch, Redirect } from 'react-router-dom';

function App() {
	const { token, setToken } = useToken();

	return (
		<div className="App">
			<header className={styles.AppHeader}>
				<BrowserRouter>
					<Switch>
						{/* <Route path="/songs/:lobbyID?">
							<SongList token={token}/>
						</Route> */}
						
						<Route path="/login">
							<Login setToken={setToken} />
						</Route>
						<Route path="/register">
							<Register />
						</Route>
						
						<PrivateRoute authed={!!token} path='/songs/:lobbyID?' component={SongList} token={token} />
						<PrivateRoute authed={!!token} path='/multiplayer/:lobbyID/:songID?' component={Multiplayer} token={token} />
						<PrivateRoute authed={!!token} path='/game/:songID/:lobbyID?' component={SongGame} token={token} />
						<PrivateRoute authed={!!token} path='/results/:gameID/:lobbyID?' component={SongEndScore} token={token} />
						
						<Route path="/confirm/:id">
							<EmailConfirm />
						</Route>
						

						<Route path="/">
							<StartPage />
						</Route>
					</Switch>
				</BrowserRouter>
			</header>
		</div>
	);
}

export default App;