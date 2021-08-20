import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import ImageService from '../services/ImageService';
import styles from './css/SongCard.module.css'
import Music from './Music';

// Song Card displayed when a song is clicked in the main page
export default class SongCard extends Component {
    constructor(props) {
        super(props);
        this.songID = props.data._id;

        this.name = props.data.name;
        this.artist = props.data.artist;
        this.length = props.data.length;
        this.creator = props.data.creator;
        this.difficulty = props.data.difficulty;
        
        this.image = ImageService.getImagebyID(this.songID);

        this.lobbyRedirect = props.lobbyID;
    }

    // What is actually displayed
    render() {
        return ( 
        <div className={styles.SongCardContainer} >
            <img className={styles.image} src={this.image} />
            <div className={styles.textwrapper}>
                <div>
                    Name: {this.name}<br />
                    Artist: {this.artist}<br />
                    Length: {this.length}<br />
                    Creator: {this.creator}<br />
                    Difficulty: {this.difficulty}<br />
                </div>
                {this.lobbyRedirect ? 
                <Link to={"/multiplayer/" + this.lobbyRedirect + "/" + this.songID+"/"}>
                    <button> Select Song </button>
                </Link>
                :<Link to={"/game/"+this.songID}>
                    <button className={styles.button}> Start </button>
                </Link>
                }
                <Music id={this.songID} />
            </div>
            
        </div>
        );
    }
}
