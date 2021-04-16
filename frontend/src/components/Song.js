import React, { Component } from 'react';
import ImageService from '../services/ImageService';
import styles from './css/Song.module.css';


class Song extends Component {
    constructor(props) {
        super(props);
        this.name = props.data.name;
        this.artist = props.data.artist;
                
        this.image = ImageService.getImage("TWICE");
    }

    // What is actually displayed
    render() {
        return ( 
        <div className={styles.SongContainer}>
            <img src={this.image} />
            <div className={styles.SongDetails}>
                {this.name}<br />
                {this.artist}
            </div>
        </div>
        );
    }
}

export default Song;