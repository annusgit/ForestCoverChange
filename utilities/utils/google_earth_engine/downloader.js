
/*
  This file will be used to download our sentinel-2 dataset as as series
*/
// start by creating an AOI first, we'll do it using a polygon instance from geometry
var pakistan = /* color: #98ff00 */ee.Geometry.Polygon(
        [[[73.65007590061041, 33.924912470572],
          [74.31637228732916, 33.924912470572],
          [74.31637228732916, 34.44520782244974],
          [73.65007590061041, 34.44520782244974]]]);

var germany = /* color: #98ff00 */ee.Geometry.Polygon(
        [[[7.7324289943328495, 49.41413955834249],
          [7.813109841500818, 49.41413955834249],
          [7.813109841500818, 49.45745251709314],
          [7.7324289943328495, 49.45745251709314]]]);

// Draw it on the map to see what we get on the map
// Map.addLayer(germany);

// some utilities

// crop function for croping out our main region of interest
function cropout_aoi(image){
  return image.clip(germany) // but this is the one that actually does the cropping
}
/*
 * Function to mask clouds using the Sentinel-2 QA band
 * @param {ee.Image} image Sentinel-2 image
 * @return {ee.Image} cloud masked Sentinel-2 image
*/
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask);//.divide(10000);
}

// rescale for our needs
function rescale(image) {
  return image.divide(4096).multiply(255).uint8(); //.cast(ee.uint8);
}

// Now define the dates we want to get data on
var date_from = ee.Date({date:'2015-07-01'});
var date_to = ee.Date({date:'2018-08-27'});

// import sentinel images for your area of interest
// and dates of interest and cloud cover as needed
var image_collection = ee.ImageCollection('COPERNICUS/S2')
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
                        .filterDate({start:date_from, end:date_to})
                        .filterBounds(germany) // locates ones close to this polygon
                        .map(cropout_aoi)
                        .map(function(image){return image.clip(germany)})
                        .map(maskS2clouds)
                        .map(rescale);
                        // .cast({'B4':ee.uint8, 'B3':ee.uint8, 'B2':ee.uint8});


// see how many images we got
var count = image_collection.size();
print('log: total images in collection =', count);


Export.video.toDrive({
                      collection: image_collection.select('B4', 'B3', 'B2'),
                      description: 'None',
                      region:germany
                    })




