import React, { useState, useEffect} from 'react';
import Music from './Music';
import ImageService from '../services/ImageService';

import axios from 'axios';

// Multiplayer lobby show song selected
export default function MultiplayerSelectedSong(props) {
    const songID = props.songID;
    const token = props.token;
    const [songDetails, setSongDetails] = useState();

    useEffect(() => {
        getSongDetails();
    }, [songID]);

    const getSongDetails = () => {
        axios.get('/songDetails', {
            params: {
                songID: songID
            },headers: {
                "Authorization": `Bearer ${token}`
            }
        }).then(response => {
            setSongDetails(response.data);
        })
    }

    if (songDetails) {
        return (
            <div style={{color: "white" }}>
                <img src={ImageService.getImagebyID(songDetails._id)} /> <br />
                <div style={{color: "white" }}> Name: {songDetails.name} </div>
                <div style={{color: "white" }}> Artist: {songDetails.artist} </div>
                <div style={{color: "white" }}> Length: {songDetails.length}</div>
                <div style={{color: "white" }}> Difficulty: {songDetails.difficulty} </div> <br />
                {<Music id={songID} />}
            </div>
        )
    }else{
        return ("LOADING");
    }
    
}

