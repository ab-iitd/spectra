STEPS 1
az login

Step 2 and 3
_______________________________________________________________________________________________

# For python agents

az acr create --resource-group myResourceGroup --name pythonagent --sku Basic --admin-enabled true

az acr build --registry pythonagent --image python-agent:latest --resource-group myResourceGroup .

_______________________________________________________________________________________________

# For image captioning agent

az acr create --resource-group myResourceGroup --name captioningagent --sku Basic --admin-enabled true

az acr build --registry captioningagent --image captioning-agent:latest --resource-group myResourceGroup .

_______________________________________________________________________________________________

# For image Detection agent

az acr create --resource-group myResourceGroup --name detectionagent --sku Basic --admin-enabled true

az acr build --registry detectionagent --image detection-agent:latest --resource-group myResourceGroup .

_______________________________________________________________________________________________

# For ocr Detection agent

az acr create --resource-group myResourceGroup --name ocragent --sku Basic --admin-enabled true

az acr build --registry ocragent --image ocr-agent:latest --resource-group myResourceGroup .
