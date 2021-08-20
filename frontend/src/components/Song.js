import React, { Component } from 'react';
import ImageService from '../services/ImageService';
import styles from './css/Song.module.css';

// Song in the list on the main page
class Song extends Component {
    constructor(props) {
        super(props);
        this.name = props.data.name;
        this.artist = props.data.artist;
        this.image = ImageService.getImagebyID(props.data._id);
    }

    // What is actually displayed
    render() {
        return ( 
        <div className={styles.SongContainer}>
            <img src={this.image} />
            <div className={styles.SongDetails}>
                Name: {this.name} <br />
                Artist: {this.artist} <br />  
            </div>
        </div>
        );
    }
}

export default Song;