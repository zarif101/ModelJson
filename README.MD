## ModelJSON
**A simple Python module to create Pytorch models from a JSON file!**

**Notes**
 - Currently only supports: Dense (Linear) layers and ReLU activation
 - See "model.json" for a template/example of how to format the JSON file
 - Please email me at zarif.azher@gmail.com if you want more info to make a pull request
 - Package does not follow all the correct formatting of an 'official' module (yet)
 - This is a work-in-progress
 - Tested with Python 3.8
 - Reach out with any questions!

 **Usage**

 - Download the repo
 - Import "get_model" from "ModelJson.py"
 - Call "get_model" with the path to your model JSON as input
 - "get_model" will return a Sequential Pytorch model with your specified layers!
 - Ex:
 ```python
 from ModelJson import get_model
 model = get_model('model.json')
 ```
 - You can test that functionality is working as expected, by running "test.py" and passing in the path to the model json file as a command line argument
 - Ex:
 ```shell
 python test.py -jsonpath model.json
 ```
