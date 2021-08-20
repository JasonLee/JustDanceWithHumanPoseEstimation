import React, { useState, useEffect} from 'react';
import styles from './css/MultiplayerPlayerBox.module.css';

import axios from 'axios';

// Player list in Multiplayer
export default function MultiplayerPlayerBox(props) {
    const username = props.username;
    const isHost = props.isHost;
    const token = props.token;

    const [playerDetails, setPlayerDetails] = useState();

    useEffect(() => {
        getUserDisplayInfo();
    }, []);

    // Gets details like username and profile pic
    const getUserDisplayInfo = () => {
        axios.get('/getUserDisplayInfo', {
            headers: {
                "Authorization" : `Bearer ${token}`
            },
            params: {
                username: username
            }
        }).then(response => {
            setPlayerDetails(response.data);
        });
    }

    if (playerDetails) {
        return (
            <div className={styles.playerWrapper}>
                <img className={styles.image} src={playerDetails.profilePic} />
                <div className={styles.user}>{playerDetails.username} </div> 
            </div>
        )
    }else{
        return ("LOADING");
    }
    
}

