import PropTypes from 'prop-types';
import { Box, Typography } from '@material-ui/core/';
import styles from "./css/TabPanel.module.css";

// Tab Panel component for Tab related display - leaderboards, multiplayer response page
export default function TabPanel(props) {
    const { children, value, index, ...other } = props;

    // Way to format child components properly to show/disappear when tab is changed
    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`simple-tabpanel-${index}`}
            aria-labelledby={`simple-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box p={3}>
                    <Typography component={'span'} className={styles.container}>{children}</Typography>
                </Box>
            )}
        </div>
    );
}

TabPanel.propTypes = {
    children: PropTypes.node,
    index: PropTypes.any.isRequired,
    value: PropTypes.any.isRequired
};
