import image from '../logo.svg';

const ImageService = {
    getImage(imageName) {
        if (imageName == "TWICE") {
            return image
        }

    }
}

export default ImageService;