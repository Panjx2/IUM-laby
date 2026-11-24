#!/usr/bin/env bash
# Zadanie 1 - Jenkins LTS w kontenerze Dockera
#
#   -d                         kontener dziala w tle
#   --name jenkins             stala nazwa: docker exec / logs / stop / start jenkins
#   -p 127.0.0.1:8080:8080     interfejs WWW pod http://localhost:8080; adres 127.0.0.1
#                              ogranicza dostep do tego komputera - samo -p 8080:8080
#                              wystawia port na wszystkich interfejsach, takze w sieci lokalnej
#   -p 127.0.0.1:50000:50000   port, na ktorym kontroler przyjmuje polaczenia agentow
#   -v jenkins_home:/var/jenkins_home
#                              wolumen nazwany na dane Jenkinsa: konfiguracja, konta,
#                              pluginy, joby i historia buildow zostaja po usunieciu kontenera
docker run -d --name jenkins \
    -p 127.0.0.1:8080:8080 -p 127.0.0.1:50000:50000 \
    -v jenkins_home:/var/jenkins_home \
    jenkins/jenkins:lts
