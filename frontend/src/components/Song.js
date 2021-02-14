import React, { Component } from 'react';


class Song extends Component {
    constructor(props) {
        super(props);
        this.id = props.id;
        this.name = props.name;
        this.artist = props.artist;

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