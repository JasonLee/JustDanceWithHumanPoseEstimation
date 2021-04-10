import React, { useState, useEffect } from 'react';
import { useParams } from "react-router-dom";
import axios from 'axios';


export default function SongEndScore(props) {
    let { gameID } = useParams();
    const [gameData, setGameData] = useState();

    useEffect(() => {    // Update the document title using the browser API    
        console.log(gameID)
        axios.post("/finish_game", {
            gameID: gameID
        })
        .then(response => {
            console.log("results returned", response.data)
            setGameData(response.data.game);
        })
        .catch(error => {
            console.log(error);
        }); 
    }, []);

    if (gameData) {
        return (
            <div>
                SongName: {gameData.songID} <br />
                Score: {gameData.score} <br />
                Accuracy: {gameData.accuracy} <br />
            </div>
            
        )
    }else{
        return("Loading");
    }
    
};
