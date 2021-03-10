import React, { useState, useEffect } from 'react';
import axios from 'axios';
import useToken from '../hooks/useToken';

function ProfileBox() {
	const { token, setToken } = useToken();
    const [ displayName, setDisplayName ] = useState(0);

    useEffect(() => {
        axios({
            method: 'get',
            url: '/user_details',
            headers: { Authorization: `Bearer ${token}` },
            data: {
                access_token: token
            } 
        }).then(res => {
            const displayName = res.data.displayName;
            console.log(displayName)
            setDisplayName(displayName);
        }).catch((error) => {
            console.log("error:", error);
        });
    }, []);

	return (
		<div>
            <p>{displayName.length > 0 ? displayName : null}</p>
		</div>
	);
}

export default ProfileBox;