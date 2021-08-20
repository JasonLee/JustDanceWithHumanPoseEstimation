import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styles from './css/PlayerBox.module.css';

// Player profile box in the top left corner
export default function PlayerBox({ token }) {
    const [playerDetails, setPlayerDetails] = useState();

    useEffect(() => {
        getUserDisplayInfo();
    }, []);

    const getUserDisplayInfo = () => {
        axios.get('/getUserDisplayInfo', {
            headers: {
                "Authorization" : `Bearer ${token}`
            }
        }).then(response => {
            setPlayerDetails(response.data);
        });
    }
    if (playerDetails) {
        return (
            <div className={styles.wrapper}>
                <img className={styles.image} src={playerDetails.profilePic} />
                <div className={styles.text}> {playerDetails.username} </div>
            </div>
        )
    }

    return "LOADING"
}

