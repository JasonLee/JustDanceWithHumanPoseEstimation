import React, { useState, useEffect } from 'react';
import { useParams } from "react-router-dom";
import axios from 'axios';
import { Link } from 'react-router-dom';
import { Button } from '@material-ui/core/';
import styles from './css/SongEndScore.module.css';
import { Tabs, Tab } from '@material-ui/core/';
import TabPanel from './TabPanel';
import useToken from '../hooks/useToken';


export default function SongEndScore(props) {
    let { gameID, lobbyID } = useParams();
    const {token, setToken} = useToken();
    const [gameData, setGameData] = useState();
    const [value, setValue] = React.useState(0);

    // Gets game results based on ID
    useEffect(() => {
        axios.get("/getResults", {
            params: {
                gameID: gameID,
                lobbyID: lobbyID
            },headers: {
                "Authorization": `Bearer ${token}`
            }
        }).then(response => {
            console.log(response.data)
            setGameData(response.data);
        })
        .catch(error => {
            console.log(error);
        }); 
    }, [gameID]);

    // Handles which tab is to be displayed
    const handleChange = (event, newValue) => {
        setValue(newValue);
    };

    const scoreBlock = (data) => {
        return (
            <div> 
                <div style={{color:"white"}} className={styles.container}>
                    <img className={styles.image} src={data.user.profilePic} />
                    <div className={styles.item}> <h1>{data.user.username}</h1> </div>
                    <div className={styles.item}>SongName: {data.songData.name}</div>
                    <div className={styles.item}>Score: {data.score}</div>
                    <div className={styles.item}>Score Breakdown:</div>
                    <div className={styles.scoreItem}><b style={{color:"Green"}}>500: </b> {data.scoreBreakdown.score500}</div>
                    <div className={styles.scoreItem}><b style={{color:"yellow"}}>300: </b> {data.scoreBreakdown.score300} </div>
                    <div className={styles.scoreItem}><b style={{color:"gray"}}>100: </b> {data.scoreBreakdown.score100} </div>
                    <div className={styles.scoreItem}><b style={{color:"red"}}>X: </b> {data.scoreBreakdown.scoreX} </div>
                </div>
               
            </div>);
    };
    
    // Results page from normal game or leaderboard
    if (gameData && lobbyID === undefined) {
        return (<div>
                    <Link to={gameID? "/songs" : `/multiplayer/${lobbyID}/`}>
                        <Button variant="contained" color="primary">BACK</Button>
                    </Link> 
                    {scoreBlock(gameData[0])}
                </div>);

    // Results page from multiplayer game - Has tabs for players
    }else if(gameData) {
        return(
            <div>
                <Link to={gameID? "/songs" : `/multiplayer/${lobbyID}/`}>
                    <Button variant="contained" color="primary">BACK</Button>
                </Link> 
                <Tabs value={value} onChange={handleChange} aria-label="simple tabs example">
                    {gameData.map((element, i) => 
                        <Tab style={{color:"white"}} key={element.gameID} label={i+1 + ". " + element.user.username} />
                    )}
                </Tabs>
                {gameData.sort((a, b) => a.score < b.score).map((element, i) =>
                    <TabPanel key={element.gameID} index={i} value={value}> 
                        {scoreBlock(element)}
                    </TabPanel>
                )}
            </div>
        );
    }else{
        return("Loading");
    }
    
};
