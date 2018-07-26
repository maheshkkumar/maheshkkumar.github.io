---
layout: post
title: A guide to deploying Machine/Deep Learning model(s) in Production
date: 2018-06-25
author: "Mahesh Kumar K"
categories: article
tags: [machine_learning]
<!-- image: machine_learning_deployment.jpeg -->
image: ml.jpeg
--- 
*Source: [XKCD](https://www.explainxkcd.com/wiki/index.php/1875:_Computers_vs_Humans)*

There are a plethora of articles on Deep Learning (DL) or Machine Learning (ML) that cover topics like data gathering, data munging, network/algorithm selection, training, validation, and evaluation. But, one of the challenging problems in today's data science is the *deployment* of the trained model in production for any consumer-centric organizations or individuals who want to make their solutions reach a wider audience.  

Most of the time, energy and resources are spent on training the model to achieve the desired results, so allocating additional time and energy to decide on the computational resources to set up the appropriate infrastructure to replicate the model for achieving similar results in a different environment (production) at scale will be a difficult task. Overall, it's a lengthy process that can easily take up months right from the decision to use DL to deploying the model.  

![Source: Algorithmia](/assets/img/machine_learning_deployment.jpeg)
*Source: [Algorithmia](https://blog.algorithmia.com/deploying-machine-learning-at-scale/)* 

---

This article tries to give a comprehensive overview of the entire process of deployment from scratch.

![Illustration of the workflow (from client API requests to server prediction responses) and you are free to use the image.](/assets/img/dl_architecture.png)
*Illustration of the workflow (from client API requests to server prediction responses). You are free to use the image.*  

**Note**: *The above image is just an illustration of a probable architecture and used primarily for learning purpose.*  

### Components  

Let's break down the above image that depicts the entire API workflow and understand every component.  

- **[Client](https://en.wikipedia.org/wiki/Client_(computing))**: The client in the architecture can be any device or a third party application that tries to request the server hosting the architecture for model predictions. Example: Facebook trying to tag your face on a newly uploaded image.  

- **[Load Balancer](https://en.wikipedia.org/wiki/Load_balancing_(computing))**: A load balancer tries to distribute the workload (requests) across multiple servers or instances in a cluster. The aim of the load balancer is to minimize the response time and maximize the throughput by avoiding the overload on any single resource. In the above image, the load balancer is public facing entity and distributes all the requests from the clients to multiple Ubuntu servers in the cluster.  

- **[Nginx](https://www.nginx.com/resources/wiki/)**: Nginx is an open-source web server but can also be used as a load balancer. Nginx has a reputation for its high performance and small memory footprint. It can thrive under heavy load by spawning worker processes, each of which can handle thousands of connections. In the image, nginx is local to one server or instance to handle all the requests from the public facing load balancer. An alternative to nginx is [Apache HTTP Server](https://httpd.apache.org/).  

- **[Gunicorn](http://docs.gunicorn.org/en/stable/)**: It is a Python Web Server Gateway Interface ([WSGI]([WSGI](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface))) HTTP server. It is ported from [Ruby's Unicorn project](https://en.wikipedia.org/wiki/Unicorn_(web_server)). It is a pre-fork worker model, which means a master creates multiple forks which are called workers to handle the requests. Since Python is not multithreaded, we try to create multiple gunicorn workers which are individual processes that have their own memory allocations to compensate the parallelism for handling requests. Gunicorn works for various Python web frameworks and a well-known alternative is [uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/).

- **[Flask](http://flask.pocoo.org/)**: It is a micro web framework written in Python. It helps us to develop application programming interface (API) or a web application that responds to the request. Other alternatives to Flask are [Django](https://www.djangoproject.com/), [Pyramid](https://trypyramid.com/), and [web2py](http://www.web2py.com/). An extension of Flask to add support for quickly building REST APIs is provided by [Flask-RESTful](https://flask-restful.readthedocs.io/en/latest/).  

- **[Keras](https://keras.io/)**: It is an open-source neural network library written in Python. It has the capability to run on top of [TensorFlow](https://www.tensorflow.org/), [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras), [Theano](http://deeplearning.net/software/theano/) or [MXNet](https://aws.amazon.com/blogs/machine-learning/apache-mxnet-incubating-adds-support-for-keras-2/). There are plenty of alternatives to Keras: [TensorFlow](https://www.tensorflow.org/), [Caffe2](https://caffe2.ai/) ([Caffe](http://caffe.berkeleyvision.org/)), [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras), [PyTorch](https://pytorch.org/), [MXNet](https://mxnet.apache.org/), [Chainer](https://chainer.org/), and [Theano](http://deeplearning.net/software/theano/) (discontinued).  

- **[Cloud Platform](https://en.wikipedia.org/wiki/Cloud_computing)**: If there is one platform that intertwines all the above-mentioned components, then it is the cloud. It is one of the primary catalysts for the proliferation in the research of Artificial Intelligence, be it Computer Vision, Natural Language Processing, Machine Learning, Machine Translation, Robotics or Medical Imaging. Cloud has made computational resources accessible to a wider audience at a reasonable cost. Few of the well-known cloud web services are [Amazon Web Services](https://aws.amazon.com/) (AWS), [Google Cloud](https://cloud.google.com/) and [Microsoft Azure](https://azure.microsoft.com/en-in/).  

---  

### Architecture Setup  

By now you should be familiar with the components mentioned in the earlier sections. In the following section, Let's understand the setup from an API perspective since this forms the base for a web application as well.  

**Note**: *This architecture setup will be based on Python.*  

### Development Setup

- **Train the model**: The first step is to train the model based on the use case using [Keras](https://keras.io/) or [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/). Make sure you do this in a [virtual environment](https://virtualenv.pypa.io/en/stable/), as it helps in isolating multiple Python environments and also it packs all the necessary dependencies into a separate folder.  

- **Build the API**: Once the model is good to go into an API, you can use [Flask](http://flask.pocoo.org/) or [Django](https://www.djangoproject.com/) to build them based on the requirement. Ideally, you have to build [Restful](https://en.wikipedia.org/wiki/Representational_state_transfer) APIs, since it helps in separating between the client and the server; improves visibility, reliability, and scalability; it is platform agnostic. Perform a thorough test to ensure the model responds with the correct predictions from the API.  

- **Web Server**: Now is the time to test the web server for the API that you have built. Gunicorn is a good choice if you have built the APIs using Flask. An example command to run the gunicorn web server.  

```python
  gunicorn --workers 1 --timeout 300 --bind 0.0.0.0:8000 api:app
  - workers (INT): The number of worker processes for handling requests.
  - timeout (INT): Workers silent for more than this many seconds are killed and restarted.
  - bind (ADDRESS): The socket to bind. [['127.0.0.1:8000']]
  - api: The main Python file containing the Flask application.
  - app: An instance of the Flask class in the main Python file 'api.py'.
```

- **Load Balancer**: You can configure nginx to handle all the test requests across all the gunicorn workers, where each worker has its own API with the DL model. Refer this [resource](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-16-04) to understand the setup between nginx and gunicorn.   

- **Load / Performance Testing**: Take a stab at [Apache Jmeter](https://jmeter.apache.org/), an open-source application designed to load test and measure performance. This will also help in understanding nginx load distribution. Alternative is [Locust](https://locust.io/).    


### Production Setup  

- **Cloud Platform**: After you have chosen the cloud service, set up a machine or instance from a standard Ubuntu image (preferably the latest  LTS version). The choice of CPU machine really depends on the DL model and the use case. Once the machine is running, setup nginx, Python virtual environment, install all the dependencies and copy the API. Finally, try running the API with the model (it might take a while to load all the model(s) based on the number of workers that are defined for gunicorn).  

- **Custom API image**: Ensure the API is working smoothly and then snapshot the instance to create the custom image that contains the API and model(s). Snapshot will preserve all the settings of the application. Reference: [AWS](https://docs.aws.amazon.com/toolkit-for-visual-studio/latest/user-guide/tkv-create-ami-from-instance.html), [Google](https://cloud.google.com/compute/docs/images/create-delete-deprecate-private-images), and [Azure](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/tutorial-custom-images).  

- **Load Balancer**: Create a load balancer from the cloud service, it can be either public or private based on the requirement. Reference: [AWS](https://docs.aws.amazon.com/elasticloadbalancing/latest/classic/elb-getting-started.html), [Google](https://cloud.google.com/load-balancing/), and [Azure](https://docs.microsoft.com/en-us/azure/load-balancer/quickstart-create-basic-load-balancer-portal).  

- **A Cluster of instances**: Use the previously created custom API image to launch a cluster of instances. Reference: [AWS](https://aws.amazon.com/premiumsupport/knowledge-center/launch-instance-custom-ami/), [Google](https://cloud.google.com/compute/docs/instances/creating-instance-with-custom-machine-type), and [Azure](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/create-vm-generalized-managed).  

- **Load Balancer for the Cluster**: Now, link the cluster of instances to a load balancer, this will ensure the load balancer to distribute work equally among all the instances. Reference: [AWS](https://docs.aws.amazon.com/elasticloadbalancing/latest/classic/elb-deregister-register-instances.html), [Google](https://cloud.google.com/compute/docs/load-balancing/http/backend-service), and [Azure](https://docs.microsoft.com/en-us/azure/load-balancer/quickstart-create-basic-load-balancer-portal).  

- **Load/Performance Test**: Just like the load/performance testing in the development, a similar procedure can be replicated in production, but now with millions of requests. Try breaking the architecture to check it's stability and reliability (not always advisable).  

- **Wrap-up**: Finally, if everything works as expected, you'll have your first production level DL architecture to serve millions of requests.  

---  

### Additional Setup (Add-ons)  

Apart from the usual setup, there are few other things to take care of to make the setup self-sustaining for the long run.  

- **Auto-scaling**: It is a feature in cloud service that helps in scaling up the instances in for the application based on the number of requests received. We can scale-out when there is a spike in the requests and scale-in when the requests have reduced. Reference: [AWS](https://docs.aws.amazon.com/autoscaling/ec2/userguide/what-is-amazon-ec2-auto-scaling.html), [Google](https://cloud.google.com/compute/docs/autoscaler/), and [Azure](https://docs.microsoft.com/en-us/azure/architecture/best-practices/auto-scaling).  

- **Application updates**: There comes a time when you have to update the application with a latest DL model or update the features of the application, but how to update all the instances without affecting the behavior of the application in production. Cloud services provide a way to perform this task in various ways and they can be very specific to a particular cloud service provider. Reference: [AWS](https://aws.amazon.com/premiumsupport/knowledge-center/auto-scaling-group-rolling-updates/), [Google](https://cloud.google.com/compute/docs/instance-groups/updating-managed-instance-groups), and [Azure](https://azure.microsoft.com/en-in/updates/auto-os-upgrades/).  

- **Continuous Integration**: It refers to the build and unit testing stages of the software release process. Every revision that is committed triggers an automated build and test. This can be used to deploy latest versions of the models into production.  

![Source: https://aws.amazon.com/devops/continuous-integration/](/assets/img/continuous_integration.png)
Source: *[AWS](https://aws.amazon.com/devops/continuous-integration/)*

---  

### Alternate platforms  

There are other systems that provide a structured way to deploy and serve models in the production and few such systems are as follows:  

- **[TensorFlow Serving](https://www.tensorflow.org/serving/)**: It is an open-source platform software library for serving machine learning models. It's primary objective based on the inference aspect of machine learning, taking trained models after training and managing their lifetimes. It has out-of-the-box support for TensorFlow models.  

![TensorFlow Serving](/assets/img/tf_serving_1.png)  
Source: *[TensorFlow Serving](https://opensource.googleblog.com/2016/02/running-your-models-in-production-with.html)*  

- **[Docker](https://github.com/floydhub/dl-docker)**: It is a container virtualization technology which behaves similarly to a light-weighted virtual machine. It provides a neat way to isolate an application with its dependencies for later use in any operating system. We  can have multiple docker images with different applications running on the same instance but without sharing the same resources.  

![Docker](/assets/img/docker_architecture.jpg)  
Source: *[Docker Architecture](https://codingpackets.com/virtualization/docker/)*  

- **[Michelangelo](https://eng.uber.com/michelangelo/)**: It is Uber's Machine Learning platform, which includes building, deploying and operating ML solutions at Uber's scale.   

![Michelangelo](/assets/img/uber_1.png)  
Source: *[Michelangelo](https://eng.uber.com/michelangelo/)*  

### Additional Resources  

- [Image classification with Keras](https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/)
- [Transfer learning in PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- Flask API for Keras model [[1](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)] [[2](https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/)]
- [Nginx and Gunicorn for Flask](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-16-04) 
- [Locust](https://locust.io/)

I hope you found this article useful and understood the overview of the deployment process of Deep/Machine Learning models from development to production.  
