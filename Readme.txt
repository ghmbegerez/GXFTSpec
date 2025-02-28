spec.md Archivo de spec 
Utilizar markdown preview y mermaid preview 

Generar PDF:
Ejecutar  
    pandoc spec.md --filter mermaid-filter -o spec.pdf
    Debe estar instalado soporte latex.
        1 Se debe instalar soporte latex, basictex o mactex y el filtro de mermaid
        brew install --cask basictex
        eval "$(/usr/libexec/path_helper)"
        2 Instalar pandoc
        brew install pandoc
        3 Instaler filtro
        npm install -g mermaid-filter

      
Directorios
    api
        api externa y servidor
            api.py api rest
            paiserver.py servidor rest
    jobs
        soporte de jobs
            jobsqueuesystem.py  
                Servidor de jobs basados en sqlite3
    
    sdk   
        api módulo
            entities.py
                Entidades principales 
            managers.py
                Adminisradores de entidades principales
            sdk.py
                Sdk principal del módulo
            finetuner.py
                Soporte de finetuning
            externaltools.py
                Soporte de librerias open source de linea de comandos.
            clisdk.py
                Cliente de linea de comandos de sdk
            utils.py
                Utilitarios


VER 0.1
    Revisando Entidades SDK y cliente SDK.
    Ajustando SPEC para que cierre con el SDK.
    