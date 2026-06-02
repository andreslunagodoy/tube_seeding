FROM debian:trixie

COPY . /app
WORKDIR /app/

RUN ./docker_install.sh

# Run Jupyter Notebook when the container launches
CMD ["/app/bin/jupyter", "notebook", "./", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
