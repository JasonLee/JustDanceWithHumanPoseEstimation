import React, { useState, useEffect } from 'react';
import { Tabs, Tab } from '@material-ui/core/';
import styles from './css/LeaderBoard.module.css'
import { Link, useLocation } from 'react-router-dom';
import TabPanel from './TabPanel';

import axios from 'axios';

// Leader Board in Song List Page
export default function LeaderBoard(props) {
    const token = props.token;
    const songID = props.songID
    let location = useLocation();

    const [userScores, setUserScores] = useState([]);
    const [globalScores, setGlobalScores] = useState([]);
    const [value, setValue] = React.useState(0);

    // Update leaderboard when page updates
    useEffect(() => {
        getScores();
    }, [songID]);

    const handleChange = (event, newValue) => {
        setValue(newValue);
    };

    const getScores = () => {
        // Get user's best scores on song
        axios.get('/getUserSongScores', {
            headers: {
                "Authorization": `Bearer ${token}`
            },
            params: {
                "songID": songID
            }
        }).then(response => {
            console.log("LEADERBOARD - user", response.data);
            setUserScores(response.data);
        });

        // Get overall global best scores on song
        axios.get('/getGlobalSongScores', {
            headers: {
                "Authorization": `Bearer ${token}`
            },
            params: {
                "songID": songID
            }
        }).then(response => {
            console.log("LEADERBOARD", response.data);
            setGlobalScores(response.data);
        });

    }

    return (
        <div className={styles.LeaderBoardWrapper}>
            <Tabs className={styles.Tabs} value={value} onChange={handleChange} aria-label="simple tabs example">
                <Tab label="My Scores" />
                <Tab label="Global Scores" />
            </Tabs>

            {userScores.sort((a, b) => a.score < b.score ? 1 : -1).map((element, i) =>
                <Link key={element.gameID} style={{ textDecoration: 'none' }} to={{pathname: `/results/${element.gameID}`, state: { from: location } }}>
                    <TabPanel className={styles.entry} key={element.gameID} index={0} value={value}> {i + 1}. {element.score} </TabPanel>
                </Link>
            )}

            {globalScores.sort((a, b) => a.score < b.score ? 1 : -1).map((element, i) =>
                <Link key={element.gameID} style={{ textDecoration: 'none' }} to={{pathname: `/results/${element.gameID}`, state: { from: location } }}>
                    <TabPanel className={styles.entry} key={element.gameID} index={1} value={value}> 
                        <div className={styles.number}>{i + 1}. </div>
                        <img className={styles.image}src={element.user.profilePic} />
                        <div className={styles.playerName}>
                            Name: <b>{element.user.username}</b> <br/>
                            Score: {element.score}
                        </div>
                    </TabPanel>
                </Link>
            )}

        </div>
    )

}

