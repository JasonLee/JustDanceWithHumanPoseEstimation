import React, { Component } from 'react';


class Song extends Component {
    constructor(props) {
        super(props);
        this.name = props.data.name;
        this.artist = props.data.artist;

    }

    // What is actually displayed
    render() {
        return ( 
        <div>
            <div>
                {this.name}
                <br />
                {this.artist}
            </div>
        </div>
        );
    }
}

export default Song;