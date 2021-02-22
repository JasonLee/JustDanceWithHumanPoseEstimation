import ImageService from '../services/ImageService'

describe('ImageService Test', () => {
    it("should return an image", () => {
        const image = ImageService.getImage("TWICE");
        // expect(image).toBeInstanceOf(Image);
        expect(image).toBe("logo.svg");
    });

});