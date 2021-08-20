import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import PropTypes from 'prop-types';
import axios from 'axios';
import { Redirect } from 'react-router-dom';
import styles from './css/Login.module.css';
import avatar from '../images/img_avatar.png';

// Login Page
export default function Login({ setToken }) {
    const [username, setUserName] = useState();
    const [password, setPassword] = useState();
    const [redirect, setRedirect] = useState();
    const [error, setError] = useState(false);
    let location = useLocation();
    const { from } = location.state || { from: { pathname: '/' } }
    
    // Submit button pressed
    const handleSubmit = async e => {
        e.preventDefault();

        axios.post("/login", {
            username: username,
            password: password
        }).then(res => {
            const token = res.data;
            setToken(token);
            setRedirect(true);
        }).catch(error => {
            setError(true);
            console.log(error);
        });
    }

    // Redirect back to page they originally wanted to go to
    if (redirect) {
        return <Redirect to={from} />;
    }

    return (
        <div className={styles.LoginWrapper}>
            <h1><b>Please Log In</b></h1>
            <form onSubmit={handleSubmit}>
                <div className={styles.ImgContainer}>
                    <img className={styles.avatar} src={avatar}/>
                </div>
                <div className={styles.Container}> 
                    <label>
                        <label><b>Username</b></label>
                        <input className={styles.input} type="text" onChange={e => setUserName(e.target.value)} required="required" />
                    </label>
                    <label>
                        <label><b>Password</b></label>
                        <input className={styles.input} type="password" onChange={e => setPassword(e.target.value)} required="required" />
                    </label>
                </div>

                {error && 
                <div className={styles.error}>
                    "You have entered an invalid username or password"
                </div>}
                <div className={styles.Container}>
                    <button className={styles.SubmitButton} type="submit">Login</button>
                    <Link to="/register" replace>
                        <b>Click here to register for an account!</b>
                    </Link>
                </div>
            </form>
        </div>
    )
}

Login.propTypes = {
    setToken: PropTypes.func.isRequired
};
